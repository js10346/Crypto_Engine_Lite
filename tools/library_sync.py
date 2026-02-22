"""Dataset library sync (Binance spot, 1D)

Part 3: Parquet support
-----------------------
This script builds/updates a local OHLCV dataset library from Binance Spot.

Why this exists
--------------
- V1 product wants *curated*, reproducible data (no user uploads).
- UI/engine only need a file path to a clean OHLCV dataset.

What it does
------------
For each coin in a registry (coins.json):
- Resolve a trading pair on Binance Spot (default quote USDT)
- Fetch daily klines (interval=1d) via Binance public REST
- Normalize into canonical schema:
    ts (ms), dt (UTC str), open, high, low, close, volume
- Write one file per coin (CSV or Parquet)
- Build/update a catalog JSON the Streamlit app can read.

Design goal
-----------
Keep the *outputs* stable so we can later swap the provider:
- Option A (today): REST klines
- Option B (later): Binance bulk data downloads

Usage
-----
  python tools/library_sync.py \
      --coins coins.json \
      --out-dir data/datasets/spot_1d \
      --catalog data/datasets/catalog.json \
      --quote USDT \
      --interval 1d \
      --format parquet

Notes
-----
- Binance returns up to 1000 candles per request; we page forward.
- This sync is incremental: if an output file exists, we only fetch newer bars.
- If writing parquet and a per-coin CSV already exists, by default we use the CSV
  as existing history (migration) and write parquet output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

BINANCE_BASE_URL = "https://api.binance.com"
MS_1D = 86_400_000


def _utc_iso_from_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _http_get_json(url: str, params: Optional[Dict[str, Any]] = None, *, timeout: int = 30) -> Any:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    symbol: str  # base asset, e.g., BTC
    quote: str   # quote asset, e.g., USDT
    pair: str    # exchange symbol id, e.g., BTCUSDT
    timeframe: str  # e.g., 1d
    file_path: str  # relative or absolute

    # Defaults must come after non-defaults (Py3.9 dataclasses requirement)
    exchange_base: Optional[str] = None  # base asset used on exchange (may differ from registry symbol)
    source: str = "binance"
    name: Optional[str] = None
    start_dt: Optional[str] = None
    end_dt: Optional[str] = None
    rows: Optional[int] = None
    updated_at: Optional[str] = None
    file_format: Optional[str] = None
    fingerprint: Optional[Dict[str, Any]] = None


def load_coin_registry(path: Path) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Return (default_quote, interval, coins_list) from registry json."""
    obj = json.loads(path.read_text(encoding="utf-8"))
    default_quote = str(obj.get("default_quote") or obj.get("quote") or "USDT")
    interval = str(obj.get("timeframe") or obj.get("interval") or "1d")

    coins = obj.get("coins")
    if not isinstance(coins, list) or not coins:
        raise ValueError("coins.json must contain a non-empty 'coins' list")

    out: List[Dict[str, Any]] = []
    for c in coins:
        if isinstance(c, str):
            out.append({"symbol": c, "enabled": True})
            continue
        if not isinstance(c, dict):
            continue
        sym = c.get("symbol") or c.get("base")
        if not sym:
            continue
        enabled = bool(c.get("enabled", True))
        out.append({**c, "symbol": str(sym).upper(), "enabled": enabled})

    out = [c for c in out if c.get("enabled", True)]
    if not out:
        raise ValueError("No enabled coins found in coins.json")

    return default_quote.upper(), interval, out


def fetch_exchange_info() -> Dict[str, Any]:
    url = f"{BINANCE_BASE_URL}/api/v3/exchangeInfo"
    return _http_get_json(url)


def build_pair_index(exchange_info: Dict[str, Any]) -> Dict[Tuple[str, str], str]:
    """Map (base, quote) -> pair symbol (e.g., ('BTC','USDT')->'BTCUSDT')."""
    out: Dict[Tuple[str, str], str] = {}
    for s in exchange_info.get("symbols", []):
        try:
            if s.get("status") != "TRADING":
                continue
            if "isSpotTradingAllowed" in s and not bool(s.get("isSpotTradingAllowed")):
                continue
            base = str(s.get("baseAsset", "")).upper()
            quote = str(s.get("quoteAsset", "")).upper()
            pair = str(s.get("symbol", "")).upper()
            if not base or not quote or not pair:
                continue
            out[(base, quote)] = pair
        except Exception:
            continue
    return out


_TICKER_RE = re.compile(r"^[A-Z0-9]{2,15}$")


def _looks_like_ticker(s: str) -> bool:
    s = (s or "").strip().upper()
    return bool(_TICKER_RE.match(s))


def _parse_csv_list(val: Optional[str]) -> List[str]:
    if val is None:
        return []
    raw = str(val).strip()
    if not raw:
        return []
    return [p.strip().upper() for p in raw.split(",") if p.strip()]


def build_base_candidates(coin: Dict[str, Any], *, allow_name_alias: bool = True) -> List[str]:
    """Return ordered unique base candidates to resolve a Binance spot pair."""
    candidates: List[str] = []

    # Explicit overrides
    for k in ("binance_base", "exchange_base"):
        v = coin.get(k)
        if isinstance(v, str) and v.strip():
            candidates.append(v.strip().upper())

    vlist = coin.get("binance_bases")
    if isinstance(vlist, list):
        for v in vlist:
            if isinstance(v, str) and v.strip():
                candidates.append(v.strip().upper())

    vlist = coin.get("aliases")
    if isinstance(vlist, list):
        for v in vlist:
            if isinstance(v, str) and v.strip():
                candidates.append(v.strip().upper())

    # Default: registry symbol
    sym = coin.get("symbol") or coin.get("base")
    if isinstance(sym, str) and sym.strip():
        candidates.append(sym.strip().upper())

    # Optional: treat name as an alias if it looks like a ticker (helps rebrands, e.g., RNDR -> RENDER)
    if allow_name_alias:
        name = coin.get("name")
        if isinstance(name, str) and _looks_like_ticker(name):
            candidates.append(name.strip().upper())

    # De-dup preserve order
    out: List[str] = []
    seen = set()
    for c in candidates:
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def build_quote_candidates(primary_quote: str, quote_fallbacks: List[str]) -> List[str]:
    quotes = [str(primary_quote).upper()]
    for q in quote_fallbacks:
        q = str(q).upper().strip()
        if q:
            quotes.append(q)
    out: List[str] = []
    seen = set()
    for q in quotes:
        if q and q not in seen:
            out.append(q)
            seen.add(q)
    return out


def resolve_pair(
    pair_index: Dict[Tuple[str, str], str],
    base_candidates: List[str],
    quote_candidates: List[str],
) -> Tuple[str, str, str]:
    """Return (pair_symbol, base_used, quote_used)."""
    tried: List[str] = []
    for base in base_candidates:
        for quote in quote_candidates:
            key = (base.upper(), quote.upper())
            tried.append(f"{key[0]}/{key[1]}")
            if key in pair_index:
                return pair_index[key], key[0], key[1]

    tried_s = ", ".join(tried[:10]) + (" ..." if len(tried) > 10 else "")
    # Helpful hint: show available quote assets for candidate bases
    hints: List[str] = []
    for b in base_candidates:
        qs = sorted({q for (bb, q) in pair_index.keys() if bb == b.upper()})
        if qs:
            hints.append(f"{b.upper()}: " + ",".join(qs[:10]) + ("..." if len(qs) > 10 else ""))
    hint_s = (" | Available: " + " ; ".join(hints)) if hints else ""
    raise ValueError(f"No Binance spot market found. Tried: {tried_s}{hint_s}")


    # Common fallback: try USDC if USDT missing.
    if quote_u == "USDT":
        alt_key = (base_u, "USDC")
        if alt_key in pair_index:
            return pair_index[alt_key], "USDC"

    raise ValueError(f"No Binance spot market found for {base_u}/{quote_u}")


def load_df(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset file type: {path}")


def _read_last_ts(path: Path) -> Optional[int]:
    """Best-effort: return last ts from an existing dataset file."""
    if not path.exists():
        return None
    try:
        suf = path.suffix.lower()
        if suf in (".parquet", ".pq"):
            df = pd.read_parquet(path, columns=["ts"])
        else:
            df = pd.read_csv(path, usecols=["ts"], dtype={"ts": "int64"})
        if df.empty or "ts" not in df.columns:
            return None
        return int(df["ts"].dropna().astype("int64").max())
    except Exception:
        return None


def fetch_klines_1d(
    pair: str,
    *,
    start_ms: int,
    end_ms: Optional[int] = None,
    limit: int = 1000,
    sleep_sec: float = 0.25,
    max_pages: int = 10_000,
) -> List[List[Any]]:
    """Fetch Binance klines (1d) paging forward."""
    out: List[List[Any]] = []
    since = int(start_ms)
    end_ms_int = int(end_ms) if end_ms is not None else None

    url = f"{BINANCE_BASE_URL}/api/v3/klines"

    for _page in range(int(max_pages)):
        params: Dict[str, Any] = {
            "symbol": pair,
            "interval": "1d",
            "startTime": since,
            "limit": int(limit),
        }
        if end_ms_int is not None:
            params["endTime"] = end_ms_int

        try:
            rows = _http_get_json(url, params=params)
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status in (418, 429):
                time.sleep(max(1.0, sleep_sec * 4))
                continue
            raise

        if not rows:
            break

        out.extend(rows)

        last_open = int(rows[-1][0])
        next_since = last_open + MS_1D
        if next_since <= since:
            break

        since = next_since
        time.sleep(max(0.0, float(sleep_sec)))

        if end_ms_int is not None and since >= end_ms_int:
            break

    if end_ms_int is not None:
        out = [r for r in out if start_ms <= int(r[0]) < end_ms_int]
    else:
        out = [r for r in out if int(r[0]) >= start_ms]

    return out


def normalize_klines(rows: List[List[Any]]) -> pd.DataFrame:
    """Binance kline rows -> canonical df."""
    if not rows:
        return pd.DataFrame(columns=["ts", "dt", "open", "high", "low", "close", "volume"])

    data = {
        "ts": [int(r[0]) for r in rows],
        "open": [float(r[1]) for r in rows],
        "high": [float(r[2]) for r in rows],
        "low": [float(r[3]) for r in rows],
        "close": [float(r[4]) for r in rows],
        "volume": [float(r[5]) for r in rows],
    }
    df = pd.DataFrame(data)
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True).astype(str)
    return df[["ts", "dt", "open", "high", "low", "close", "volume"]]


def _ensure_canonical(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ts" not in out.columns:
        raise ValueError("existing dataset missing ts")
    if "dt" not in out.columns:
        out["dt"] = pd.to_datetime(out["ts"], unit="ms", utc=True).astype(str)

    out["ts"] = pd.to_numeric(out["ts"], errors="coerce")
    out = out.dropna(subset=["ts"]).copy()
    out["ts"] = out["ts"].astype("int64")
    out = out.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    out["dt"] = pd.to_datetime(out["ts"], unit="ms", utc=True).astype(str)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = 0.0

    return out[["ts", "dt", "open", "high", "low", "close", "volume"]]


def merge_existing(existing_path: Path, fresh: pd.DataFrame) -> pd.DataFrame:
    if not existing_path.exists():
        return fresh
    try:
        old = load_df(existing_path)
        old = _ensure_canonical(old)
    except Exception:
        return fresh

    combo = pd.concat([old, fresh], ignore_index=True)
    combo = _ensure_canonical(combo)
    return combo


def write_dataset(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()
    if fmt == "csv":
        df.to_csv(path, index=False)
        return
    if fmt in ("parquet", "pq"):
        try:
            df.to_parquet(path, index=False)
        except Exception as e:
            raise RuntimeError(
                "Failed to write parquet. Install pyarrow (recommended) or fastparquet. "
                f"Original error: {e}"
            )
        return
    raise ValueError(f"Unsupported --format: {fmt}")


def build_catalog_entry(
    *,
    base_symbol: str,
    quote: str,
    pair: str,
    exchange_base: Optional[str] = None,
    interval: str,
    rel_path: str,
    df: pd.DataFrame,
    file_format: str,
    source: str = "binance",
    name: Optional[str] = None,
) -> DatasetSpec:
    start_dt = str(df["dt"].iloc[0]) if len(df) else None
    end_dt = str(df["dt"].iloc[-1]) if len(df) else None
    rows = int(len(df))

    updated_at = datetime.now(tz=timezone.utc).isoformat()
    ds_id = f"{source}_spot_{interval}_{pair}".lower()

    return DatasetSpec(
        id=ds_id,
        symbol=base_symbol.upper(),
        quote=quote.upper(),
        pair=pair,
        exchange_base=exchange_base,
        timeframe=interval,
        file_path=rel_path,
        source=source,
        name=name,
        start_dt=start_dt,
        end_dt=end_dt,
        rows=rows,
        updated_at=updated_at,
        file_format=file_format,
    )


def write_catalog_json(entries: List[DatasetSpec], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: List[Dict[str, Any]] = []
    for e in entries:
        payload.append(
            {
                "id": e.id,
                "symbol": e.symbol,
                "name": e.name or e.symbol,
                "timeframe": "1D" if e.timeframe == "1d" else e.timeframe,
                "start_dt": e.start_dt,
                "end_dt": e.end_dt,
                "rows": e.rows,
                "source": e.source,
                "quote": e.quote,
                "pair": e.pair,
                "exchange_base": e.exchange_base,
                "file_path": e.file_path,
                "file_format": e.file_format,
                "updated_at": e.updated_at,
                "fingerprint": e.fingerprint,
            }
        )

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Sync Binance spot daily OHLCV library")
    ap.add_argument("--coins", required=True, help="Path to coins.json")
    ap.add_argument(
        "--out-dir",
        default="data/datasets/spot_1d",
        help="Output directory for per-coin files (default: data/datasets/spot_1d)",
    )
    ap.add_argument(
        "--catalog",
        default="data/datasets/catalog.json",
        help="Catalog output path (default: data/datasets/catalog.json)",
    )
    ap.add_argument("--quote", default=None, help="Override quote asset (default from coins.json or USDT)")
    ap.add_argument(
        "--quote-fallbacks",
        default="FDUSD,USDC,TUSD",
        help="Comma-separated quote assets to try if the primary quote has no spot pair (default: FDUSD,USDC,TUSD)",
    )
    ap.add_argument(
        "--no-name-alias",
        action="store_true",
        help="Do not try coin 'name' as a base-asset alias when resolving a Binance spot pair (useful for rebrands).",
    )
    ap.add_argument("--interval", default=None, help="Override interval/timeframe (default from coins.json or 1d)")
    ap.add_argument("--sleep", type=float, default=0.25, help="Sleep seconds between API pages")
    ap.add_argument("--start", default=None, help="Optional ISO start date (YYYY-MM-DD) for initial pull")
    ap.add_argument("--end", default=None, help="Optional ISO end date (YYYY-MM-DD) for pull (exclusive)")
    ap.add_argument("--force-full", action="store_true", help="Ignore existing files; refetch full history")
    ap.add_argument("--dry-run", action="store_true", help="Do not write files; just print actions")
    ap.add_argument("--fingerprint", action="store_true", help="Compute sha256 fingerprints (slower)")
    ap.add_argument(
        "--format",
        default="csv",
        choices=["csv", "parquet"],
        help="Output file format (csv or parquet). Default: csv",
    )
    ap.add_argument(
        "--no-migrate-from-csv",
        action="store_true",
        help="Disable automatic migration: when writing parquet and the parquet file is missing but a CSV exists, "
        "by default the script uses the CSV as existing history and writes parquet output.",
    )
    ap.add_argument(
        "--delete-migrated-csv",
        action="store_true",
        help="If --format parquet and a CSV was used for migration, delete the old CSV after successful write.",
    )
    args = ap.parse_args()

    repo_root = Path.cwd().resolve()

    coins_path = Path(args.coins)
    default_quote, registry_interval, coins = load_coin_registry(coins_path)

    quote = str(args.quote).upper() if args.quote else default_quote
    interval = str(args.interval).lower() if args.interval else str(registry_interval).lower()
    if interval != "1d":
        raise ValueError("This script currently supports interval=1d only (daily)")

    out_dir = (repo_root / args.out_dir).resolve()
    catalog_path = (repo_root / args.catalog).resolve()

    fmt = str(args.format).lower()
    out_ext = "csv" if fmt == "csv" else "parquet"

    start_ms_override: Optional[int] = None
    end_ms_override: Optional[int] = None

    if args.start:
        start_ms_override = int(pd.to_datetime(args.start, utc=True).value // 1_000_000)
    if args.end:
        end_ms_override = int(pd.to_datetime(args.end, utc=True).value // 1_000_000)

    print(f"Coins registry: {coins_path}")
    print(f"Output dir:     {out_dir}")
    print(f"Catalog:        {catalog_path}")
    print(f"Source:         Binance spot")
    print(f"Quote:          {quote}  (fallbacks: {args.quote_fallbacks})")
    print(f"Interval:       {interval}")
    print(f"Format:         {fmt}")

    ex_info = fetch_exchange_info()
    pair_index = build_pair_index(ex_info)

    catalog_entries: List[DatasetSpec] = []

    for coin in coins:
        base = str(coin.get("symbol")).upper()
        name = coin.get("name")
        enabled = coin.get("enabled", True)
        if enabled is False:
            print(f"[SKIP] {base}: disabled in coins registry")
            continue

        base_candidates = build_base_candidates(coin, allow_name_alias=(not args.no_name_alias))
        quote_candidates = build_quote_candidates(quote, _parse_csv_list(args.quote_fallbacks))

        try:
            pair, base_used, quote_used = resolve_pair(pair_index, base_candidates, quote_candidates)
        except Exception as e:
            print(f"[SKIP] {base}: {e}")
            continue

        out_file = out_dir / f"{base}.{out_ext}"
        existing_path = out_file

        migrated_csv: Optional[Path] = None
        if fmt == "parquet" and not args.no_migrate_from_csv and not out_file.exists():
            candidate_csv = out_dir / f"{base}.csv"
            if candidate_csv.exists():
                existing_path = candidate_csv
                migrated_csv = candidate_csv

        # Decide start time
        if args.force_full or not existing_path.exists():
            start_ms = start_ms_override if start_ms_override is not None else 0
        else:
            last_ts = _read_last_ts(existing_path)
            if last_ts is None:
                start_ms = start_ms_override if start_ms_override is not None else 0
            else:
                start_ms = int(last_ts + MS_1D)
                if start_ms_override is not None:
                    start_ms = max(start_ms, int(start_ms_override))

        end_ms = end_ms_override

        if start_ms == 0 and start_ms_override is not None:
            start_ms = int(start_ms_override)

        if start_ms <= 0:
            # 2017-01-01 UTC (Binance spot inception-ish) — safe default
            start_ms = int(pd.to_datetime("2017-01-01", utc=True).value // 1_000_000)

        print(
            f"[SYNC] {base_used}/{quote_used} ({pair}) -> {out_file.name} "
            f"start={_utc_iso_from_ms(start_ms)}"
        )

        if args.dry_run:
            continue

        rows = fetch_klines_1d(
            pair,
            start_ms=start_ms,
            end_ms=end_ms,
            sleep_sec=float(args.sleep),
        )
        fresh = normalize_klines(rows)

        if fresh.empty and existing_path.exists() and not args.force_full:
            try:
                df = _ensure_canonical(load_df(existing_path))
            except Exception:
                df = fresh
        else:
            if existing_path.exists() and not args.force_full:
                df = merge_existing(existing_path, fresh)
            else:
                df = fresh

        write_dataset(df, out_file, fmt)

        if migrated_csv is not None and args.delete_migrated_csv:
            try:
                migrated_csv.unlink(missing_ok=True)
                print(f"  deleted migrated csv: {migrated_csv.name}")
            except Exception:
                pass

        # Build catalog entry with relative path
        try:
            rel_path = str(out_file.relative_to(repo_root).as_posix())
        except Exception:
            rel_path = str(out_file.as_posix())

        entry = build_catalog_entry(
            base_symbol=base,
            quote=quote_used,
            pair=pair,
            exchange_base=base_used,
            interval=interval,
            rel_path=rel_path,
            df=df,
            file_format=fmt,
            name=name,
        )

        if args.fingerprint:
            try:
                entry = DatasetSpec(
                    **{
                        **entry.__dict__,
                        "fingerprint": {
                            "sha256": _sha256_file(out_file),
                            "bytes": int(out_file.stat().st_size),
                            "mtime": float(out_file.stat().st_mtime),
                        },
                    }
                )
            except Exception:
                pass

        catalog_entries.append(entry)

    if not args.dry_run:
        write_catalog_json(catalog_entries, catalog_path)
        print(f"Wrote catalog: {catalog_path}  entries={len(catalog_entries)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
