#!/usr/bin/env python3
"""
Rebuild a dataset catalog from a library directory.

Why:
- Your UI reads a lightweight catalog (JSON) to list datasets quickly.
- Sometimes you want to regenerate it from disk (after adding/removing files).

This script scans a directory for OHLCV datasets (CSV/Parquet) and writes
a catalog JSON compatible with the dataset picker UI.

Default assumptions:
- Spot OHLCV, daily candles (1D)
- One file per coin/symbol
- Canonical columns: ts, dt, open, high, low, close, volume
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


CANON_COLS = ["ts", "dt", "open", "high", "low", "close", "volume"]


def repo_root() -> Path:
    # If placed at repo/tools/library_catalog.py -> parents[1] is repo root.
    # Otherwise fallback to CWD.
    p = Path(__file__).resolve()
    if len(p.parents) >= 2:
        return p.parents[1]
    return Path.cwd().resolve()


def load_df(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def normalize_minimal(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure at least ts/dt exist and are parseable for meta extraction."""
    cols = {c.lower(): c for c in df.columns}
    # Accept a few aliases
    ts_col = cols.get("ts") or cols.get("timestamp") or cols.get("open_time")
    dt_col = cols.get("dt") or cols.get("datetime") or cols.get("date")
    if ts_col is None and dt_col is None:
        raise ValueError("Missing both ts and dt columns.")

    out = df.copy()

    if ts_col is None:
        # build from dt
        dt = pd.to_datetime(out[dt_col], utc=True, errors="coerce")
        if dt.isna().all():
            raise ValueError("dt exists but could not be parsed.")
        out["ts"] = (dt.view("int64") // 1_000_000).astype("int64")
    else:
        out["ts"] = pd.to_numeric(out[ts_col], errors="coerce").astype("Int64")

    if dt_col is None:
        out["dt"] = pd.to_datetime(out["ts"].astype("int64"), unit="ms", utc=True, errors="coerce").astype(str)
    else:
        out["dt"] = pd.to_datetime(out[dt_col], utc=True, errors="coerce").astype(str)

    out = out.dropna(subset=["ts"])
    out = out.sort_values("ts")
    return out


def infer_symbol_from_filename(path: Path) -> str:
    # BTC.csv -> BTC
    return path.stem.split(".")[0].split("_")[0].upper()


def make_entry(
    *,
    ds_path: Path,
    symbol: str,
    timeframe: str,
    source: str,
    quote: str,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    df = load_df(ds_path)
    df = normalize_minimal(df)
    rows = int(len(df))
    if rows == 0:
        start_dt = end_dt = None
    else:
        start_dt = str(df["dt"].iloc[0])
        end_dt = str(df["dt"].iloc[-1])

    rr = repo_root()
    rel = str(ds_path.resolve().relative_to(rr)) if ds_path.resolve().is_relative_to(rr) else str(ds_path.resolve())
    pair = f"{symbol}/{quote}"

    # Use mtime as updated_at
    try:
        updated_at = pd.to_datetime(ds_path.stat().st_mtime, unit="s", utc=True).isoformat()
    except Exception:
        updated_at = None

    entry = {
        "id": f"spot_{timeframe}_{source}_{symbol}{quote}".lower(),
        "symbol": symbol,
        "name": name or symbol,
        "timeframe": "1D" if timeframe.lower() == "1d" else timeframe,
        "start_dt": start_dt,
        "end_dt": end_dt,
        "rows": rows,
        "source": source,
        "quote": quote,
        "pair": pair,
        "file_path": rel,
        "updated_at": updated_at,
    }
    return entry


def scan_dir(data_dir: Path) -> List[Path]:
    out: List[Path] = []
    for suf in ("*.csv", "*.parquet", "*.pq"):
        out.extend(sorted(data_dir.rglob(suf)))
    # Drop obvious non-datasets
    out = [p for p in out if p.name.lower() not in ("catalog.json",)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild dataset catalog by scanning a directory")
    ap.add_argument("--data-dir", required=True, help="Directory containing datasets (csv/parquet)")
    ap.add_argument("--out", required=True, help="Where to write catalog.json")
    ap.add_argument("--timeframe", default="1d", help="Default timeframe label (e.g., 1d)")
    ap.add_argument("--source", default="binance", help="Source/provider label")
    ap.add_argument("--quote", default="USDT", help="Quote currency label")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = scan_dir(data_dir)
    if not paths:
        print(f"[catalog] No datasets found under: {data_dir}")
        out_path.write_text("[]", encoding="utf-8")
        return 0

    entries: List[Dict[str, Any]] = []
    errors: List[str] = []
    for p in paths:
        sym = infer_symbol_from_filename(p)
        try:
            entries.append(
                make_entry(
                    ds_path=p,
                    symbol=sym,
                    timeframe=args.timeframe,
                    source=args.source,
                    quote=args.quote,
                )
            )
        except Exception as e:
            errors.append(f"{p}: {e}")

    # Sort nicely
    entries.sort(key=lambda d: d.get("symbol", ""))

    out_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"[catalog] Wrote {len(entries)} entries -> {out_path}")

    if errors:
        print("\n[catalog] Errors:")
        for e in errors:
            print(" -", e)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
