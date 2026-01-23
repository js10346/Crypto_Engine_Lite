# tools/fetch_usdm_perp_1m_history.py

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from ccxt.base.errors import (
    BadRequest,
    DDoSProtection,
    ExchangeNotAvailable,
    NetworkError,
    RateLimitExceeded,
    RequestTimeout,
)

MS_1M = 60_000


def floor_to_minute_ms(ms: int) -> int:
    return (ms // MS_1M) * MS_1M


def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def iso_to_ms(iso: str) -> int:
    ts = pd.to_datetime(iso, utc=True)
    return int(ts.value // 1_000_000)


def parse_end_ms(end_iso: Optional[str]) -> int:
    if not end_iso:
        now_ms = int(time.time() * 1000)
        return floor_to_minute_ms(now_ms)
    return floor_to_minute_ms(iso_to_ms(end_iso))


def make_exchange() -> ccxt.Exchange:
    ex = ccxt.binanceusdm(
        {
            "enableRateLimit": True,
            "options": {
                "adjustForTimeDifference": False,
            },
        }
    )
    ex.load_markets()
    return ex


def resolve_market(ex: ccxt.Exchange, symbol: str) -> Tuple[str, str]:
    """
    Returns (ccxt_symbol, binance_symbol_id), where binance_symbol_id is like "ETHUSDT".
    """
    if symbol in ex.markets:
        m = ex.market(symbol)
        return symbol, str(m["id"])

    alt = symbol + ":USDT"
    if alt in ex.markets:
        m = ex.market(alt)
        return alt, str(m["id"])

    suggestions = [s for s in ex.markets.keys() if "ETH/USDT" in s][:10]
    raise ValueError(f"Symbol '{symbol}' not found. Try one of: {suggestions}")


def _sleep_polite(seconds: float) -> None:
    time.sleep(max(0.0, float(seconds)))


def _ensure_raw_endpoint(ex: ccxt.Exchange, name: str) -> None:
    if not hasattr(ex, name):
        raise AttributeError(
            f"ccxt exchange missing method '{name}'. "
            "Your ccxt version may not expose this endpoint."
        )


def _retryable_fetch(fn, *, max_retries: int, sleep_sec: float) -> Any:
    """
    Retry on transient errors only. Do not retry BadRequest (400) errors.
    """
    last_err: Optional[Exception] = None
    for attempt in range(int(max_retries) + 1):
        try:
            return fn()
        except Exception as e:
            if isinstance(e, BadRequest):
                raise

            transient = isinstance(
                e,
                (
                    RateLimitExceeded,
                    DDoSProtection,
                    NetworkError,
                    RequestTimeout,
                    ExchangeNotAvailable,
                ),
            )
            if not transient:
                raise

            last_err = e
            backoff = sleep_sec * (2**attempt)
            _sleep_polite(backoff)

    raise last_err  # type: ignore[misc]


def fetch_last_ohlcv_1m(
    ex: ccxt.Exchange,
    ccxt_symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    limit: int,
    sleep_sec: float,
    max_retries: int,
) -> List[List[Any]]:
    out: List[List[Any]] = []
    since = int(start_ms)

    while since < end_ms:

        def _call():
            return ex.fetch_ohlcv(
                ccxt_symbol, timeframe="1m", since=since, limit=int(limit)
            )

        candles = _retryable_fetch(_call, max_retries=max_retries, sleep_sec=sleep_sec)
        if not candles:
            break

        out.extend(candles)

        last_ts = int(candles[-1][0])
        next_since = last_ts + MS_1M
        if next_since <= since:
            break

        since = next_since
        _sleep_polite(sleep_sec)

    out = [c for c in out if start_ms <= int(c[0]) < end_ms]
    return out


def fetch_mark_klines_1m(
    ex: ccxt.Exchange,
    binance_symbol_id: str,
    start_ms: int,
    end_ms: int,
    *,
    limit: int,
    sleep_sec: float,
    max_retries: int,
) -> List[List[Any]]:
    _ensure_raw_endpoint(ex, "fapiPublicGetMarkPriceKlines")

    out: List[List[Any]] = []
    since = int(start_ms)

    while since < end_ms:
        params = {
            "symbol": binance_symbol_id,
            "interval": "1m",
            "startTime": since,
            "endTime": end_ms,
            "limit": int(limit),
        }

        def _call():
            return ex.fapiPublicGetMarkPriceKlines(params)

        rows = _retryable_fetch(_call, max_retries=max_retries, sleep_sec=sleep_sec)
        if not rows:
            break

        out.extend(rows)

        last_ts = int(rows[-1][0])
        next_since = last_ts + MS_1M
        if next_since <= since:
            break

        since = next_since
        _sleep_polite(sleep_sec)

    out = [r for r in out if start_ms <= int(r[0]) < end_ms]
    return out


def fetch_index_klines_1m(
    ex: ccxt.Exchange,
    binance_symbol_id: str,
    start_ms: int,
    end_ms: int,
    *,
    limit: int,
    sleep_sec: float,
    max_retries: int,
) -> List[List[Any]]:
    _ensure_raw_endpoint(ex, "fapiPublicGetIndexPriceKlines")

    out: List[List[Any]] = []
    since = int(start_ms)

    while since < end_ms:
        params = {
            # Binance USDT-M indexPriceKlines uses 'pair'
            "pair": binance_symbol_id,
            "interval": "1m",
            "startTime": since,
            "endTime": end_ms,
            "limit": int(limit),
        }

        def _call():
            return ex.fapiPublicGetIndexPriceKlines(params)

        rows = _retryable_fetch(_call, max_retries=max_retries, sleep_sec=sleep_sec)
        if not rows:
            break

        out.extend(rows)

        last_ts = int(rows[-1][0])
        next_since = last_ts + MS_1M
        if next_since <= since:
            break

        since = next_since
        _sleep_polite(sleep_sec)

    out = [r for r in out if start_ms <= int(r[0]) < end_ms]
    return out


def fetch_funding_rate_history(
    ex: ccxt.Exchange,
    binance_symbol_id: str,
    start_ms: int,
    end_ms: int,
    *,
    limit: int,
    sleep_sec: float,
    max_retries: int,
) -> List[Dict[str, Any]]:
    _ensure_raw_endpoint(ex, "fapiPublicGetFundingRate")

    out: List[Dict[str, Any]] = []
    since = int(start_ms)

    while since < end_ms:
        params = {
            "symbol": binance_symbol_id,
            "startTime": since,
            "endTime": end_ms,
            "limit": int(limit),
        }

        def _call():
            return ex.fapiPublicGetFundingRate(params)

        rows = _retryable_fetch(_call, max_retries=max_retries, sleep_sec=sleep_sec)
        if not rows:
            break

        out.extend(rows)

        last_t = int(rows[-1]["fundingTime"])
        next_since = last_t + 1
        if next_since <= since:
            break

        since = next_since
        _sleep_polite(sleep_sec)

    out = [r for r in out if start_ms <= int(r.get("fundingTime", 0)) < end_ms]
    return out


def to_df_last(rows: List[List[Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("Int64")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ts"]).copy()
    df["ts"] = df["ts"].astype("int64")
    return df


def to_df_ohlc_from_klines(rows: List[List[Any]], prefix: str) -> pd.DataFrame:
    """
    Mark/index kline arrays contain many fields. We only need:
    [openTime, open, high, low, close]
    """
    if not rows:
        return pd.DataFrame(
            columns=[
                "ts",
                f"{prefix}_open",
                f"{prefix}_high",
                f"{prefix}_low",
                f"{prefix}_close",
            ]
        )

    df = pd.DataFrame(rows)
    if df.shape[1] < 5:
        return pd.DataFrame(
            columns=[
                "ts",
                f"{prefix}_open",
                f"{prefix}_high",
                f"{prefix}_low",
                f"{prefix}_close",
            ]
        )

    df = df.iloc[:, :5]
    df.columns = ["ts", "open", "high", "low", "close"]
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("Int64")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ts"]).copy()
    df["ts"] = df["ts"].astype("int64")

    return df.rename(
        columns={
            "open": f"{prefix}_open",
            "high": f"{prefix}_high",
            "low": f"{prefix}_low",
            "close": f"{prefix}_close",
        }
    )


def build_minute_index(start_ms: int, end_ms: int) -> pd.DataFrame:
    ts = np.arange(start_ms, end_ms, MS_1M, dtype=np.int64)
    df = pd.DataFrame({"ts": ts})
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def validate_alignment(df: pd.DataFrame, start_ms: int, end_ms: int, col: str) -> Dict[str, Any]:
    report: Dict[str, Any] = {"rows": int(len(df))}
    if df.empty:
        report["missing_minutes"] = int((end_ms - start_ms) // MS_1M)
        return report

    report["duplicate_ts"] = int(df["ts"].duplicated().sum())
    report["bad_timestamp_alignment"] = int((df["ts"] % MS_1M != 0).sum())

    ts = np.unique(df["ts"].astype(np.int64).to_numpy())
    expected = int((end_ms - start_ms) // MS_1M)
    report["missing_minutes"] = int(max(0, expected - int(len(ts))))

    if col in df.columns:
        report[f"{col}_nan"] = int(pd.to_numeric(df[col], errors="coerce").isna().sum())

    return report


def month_ranges_utc(start_ms: int, end_ms: int) -> Iterable[Tuple[int, int, str]]:
    """
    Yields (start_ms, end_ms, tag) month-by-month.
    tag is YYYYMM.
    """
    start_dt = pd.to_datetime(start_ms, unit="ms", utc=True)
    end_dt = pd.to_datetime(end_ms, unit="ms", utc=True)

    cur = start_dt
    while cur < end_dt:
        month_start = cur
        next_month = (month_start + pd.offsets.MonthBegin(1)).normalize()
        # Keep timezone
        next_month = next_month.tz_localize("UTC") if next_month.tzinfo is None else next_month

        month_end = min(next_month, end_dt)
        s_ms = int(month_start.value // 1_000_000)
        e_ms = int(month_end.value // 1_000_000)
        tag = month_start.strftime("%Y%m")
        yield s_ms, e_ms, tag
        cur = month_end


def merge_month(
    *,
    start_ms: int,
    end_ms: int,
    df_last: pd.DataFrame,
    df_mark: pd.DataFrame,
    df_index: pd.DataFrame,
    df_funding: pd.DataFrame,
) -> pd.DataFrame:
    base = build_minute_index(start_ms, end_ms)

    df = base.merge(df_last, on="ts", how="left")
    df = df.merge(df_mark, on="ts", how="left")
    df = df.merge(df_index, on="ts", how="left")
    df = df.merge(df_funding, on="ts", how="left")

    df["funding_rate"] = pd.to_numeric(df.get("funding_rate"), errors="coerce").fillna(0.0).astype(float)
    df["is_funding_event"] = (df["funding_rate"].abs() > 0).astype(bool)

    # Ensure consistent column order
    cols = [
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "mark_open",
        "mark_high",
        "mark_low",
        "mark_close",
        "index_open",
        "index_high",
        "index_low",
        "index_close",
        "funding_rate",
        "is_funding_event",
        "dt",
    ]
    # Some merges may not include all columns if upstream empty; add missing
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]

    return df


def write_report(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch Binance USDT-M perp 1m history (month-by-month)")
    ap.add_argument("--symbol", default="ETH/USDT", help="CCXT symbol (default ETH/USDT)")
    ap.add_argument("--start", default="2023-01-01T00:00:00Z", help="Start ISO UTC")
    ap.add_argument("--end", default=None, help="End ISO UTC (default: now, floored to minute)")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--tmp-dir", default="data/_tmp_ethusdt_2023_to_now_parts")
    ap.add_argument("--out", default="data/ethusdt_usdm_perp_1m_2023_to_now.parquet")
    ap.add_argument("--report", default="data/ethusdt_usdm_perp_1m_2023_to_now_report.json")
    ap.add_argument("--force", action="store_true", help="Re-download and overwrite month parts")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any NaNs appear in merged last/mark/index closes for a month",
    )
    args = ap.parse_args()

    start_ms = floor_to_minute_ms(iso_to_ms(args.start))
    end_ms = parse_end_ms(args.end)

    if end_ms <= start_ms:
        raise ValueError("end must be after start")

    print(f"Window UTC: {ms_to_iso(start_ms)}  ->  {ms_to_iso(end_ms)} (exclusive end)")

    ex = make_exchange()
    ccxt_symbol, sym_id = resolve_market(ex, args.symbol)
    print(f"Exchange: {ex.id} | CCXT symbol: {ccxt_symbol} | Binance symbol: {sym_id}")

    tmp_dir = Path(args.tmp_dir).resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    months: List[Dict[str, Any]] = []
    total_expected = int((end_ms - start_ms) // MS_1M)
    total_funding_events = 0

    # Fetch + write month parts
    for s_ms, e_ms, tag in month_ranges_utc(start_ms, end_ms):
        part_path = tmp_dir / f"{sym_id}_merged_{tag}.parquet"
        if part_path.exists() and not args.force:
            print(f"[SKIP] {tag} exists: {part_path.name}")
            months.append({"tag": tag, "status": "skipped", "path": str(part_path)})
            continue

        print(f"[FETCH] {tag}  {ms_to_iso(s_ms)} -> {ms_to_iso(e_ms)}")
        t0 = time.time()

        last_raw = fetch_last_ohlcv_1m(
            ex,
            ccxt_symbol,
            s_ms,
            e_ms,
            limit=int(args.limit),
            sleep_sec=float(args.sleep),
            max_retries=int(args.retries),
        )
        t1 = time.time()

        mark_raw = fetch_mark_klines_1m(
            ex,
            sym_id,
            s_ms,
            e_ms,
            limit=int(args.limit),
            sleep_sec=float(args.sleep),
            max_retries=int(args.retries),
        )
        t2 = time.time()

        index_raw = fetch_index_klines_1m(
            ex,
            sym_id,
            s_ms,
            e_ms,
            limit=int(args.limit),
            sleep_sec=float(args.sleep),
            max_retries=int(args.retries),
        )
        t3 = time.time()

        funding_raw = fetch_funding_rate_history(
            ex,
            sym_id,
            s_ms,
            e_ms,
            limit=int(args.limit),
            sleep_sec=float(args.sleep),
            max_retries=int(args.retries),
        )
        t4 = time.time()

        df_last = to_df_last(last_raw).sort_values("ts").drop_duplicates("ts", keep="last")
        df_mark = to_df_ohlc_from_klines(mark_raw, "mark").sort_values("ts").drop_duplicates("ts", keep="last")
        df_index = to_df_ohlc_from_klines(index_raw, "index").sort_values("ts").drop_duplicates("ts", keep="last")

        df_funding = pd.DataFrame(funding_raw)
        if not df_funding.empty:
            df_funding["ts"] = pd.to_numeric(df_funding["fundingTime"], errors="coerce")
            df_funding["funding_rate"] = pd.to_numeric(df_funding["fundingRate"], errors="coerce")
            df_funding = df_funding.dropna(subset=["ts"]).copy()
            df_funding["ts"] = df_funding["ts"].astype("int64")
            df_funding = df_funding[["ts", "funding_rate"]].drop_duplicates("ts", keep="last")
        else:
            df_funding = pd.DataFrame(columns=["ts", "funding_rate"])

        total_funding_events += int(len(df_funding))

        merged = merge_month(
            start_ms=s_ms,
            end_ms=e_ms,
            df_last=df_last,
            df_mark=df_mark,
            df_index=df_index,
            df_funding=df_funding,
        )

        expected_minutes = int((e_ms - s_ms) // MS_1M)
        missing = {
            "last_close_nan": int(pd.to_numeric(merged["close"], errors="coerce").isna().sum()),
            "mark_close_nan": int(pd.to_numeric(merged["mark_close"], errors="coerce").isna().sum()),
            "index_close_nan": int(pd.to_numeric(merged["index_close"], errors="coerce").isna().sum()),
        }

        if args.strict and any(v > 0 for v in missing.values()):
            raise RuntimeError(f"[{tag}] missing data detected: {missing}")

        merged.to_parquet(part_path, index=False)

        info = {
            "tag": tag,
            "status": "written",
            "path": str(part_path),
            "start_utc": ms_to_iso(s_ms),
            "end_utc_exclusive": ms_to_iso(e_ms),
            "expected_minutes": expected_minutes,
            "raw_counts": {
                "last_rows_raw": int(len(last_raw)),
                "mark_rows_raw": int(len(mark_raw)),
                "index_rows_raw": int(len(index_raw)),
                "funding_events": int(len(df_funding)),
            },
            "series_validation": {
                "last": validate_alignment(df_last, s_ms, e_ms, "close"),
                "mark": validate_alignment(df_mark, s_ms, e_ms, "mark_close"),
                "index": validate_alignment(df_index, s_ms, e_ms, "index_close"),
            },
            "merged_missing": missing,
            "fetch_sec": {
                "last_ohlcv": float(t1 - t0),
                "mark_klines": float(t2 - t1),
                "index_klines": float(t3 - t2),
                "funding": float(t4 - t3),
                "total": float(t4 - t0),
            },
        }
        months.append(info)
        print(f"[OK] {tag} wrote {expected_minutes} rows  missing={missing}")

    # Build final big parquet from month parts
    part_files = sorted(tmp_dir.glob(f"{sym_id}_merged_*.parquet"))
    if not part_files:
        raise RuntimeError(f"No part files found in {tmp_dir}")

    print(f"\nBuilding final parquet from {len(part_files)} month parts...")
    if out_path.exists():
        out_path.unlink()

    writer: Optional[pq.ParquetWriter] = None
    total_rows = 0

    for p in part_files:
        df_part = pd.read_parquet(p)
        table = pa.Table.from_pandas(df_part, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")

        writer.write_table(table)
        total_rows += int(table.num_rows)

    if writer is not None:
        writer.close()

    final_report = {
        "exchange": ex.id,
        "ccxt_symbol": ccxt_symbol,
        "binance_symbol": sym_id,
        "timeframe": "1m",
        "start_utc": ms_to_iso(start_ms),
        "end_utc_exclusive": ms_to_iso(end_ms),
        "expected_minutes_total": int(total_expected),
        "written_rows_total": int(total_rows),
        "funding_events_total": int(total_funding_events),
        "tmp_dir": str(tmp_dir),
        "out_parquet": str(out_path),
        "months": months,
    }
    write_report(report_path, final_report)

    print("\nSaved:")
    print(f"  {out_path}")
    print(f"  {report_path}")
    print(f"\nRows written: {total_rows} (expected {total_expected})")
    print(f"Funding events total: {total_funding_events}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())