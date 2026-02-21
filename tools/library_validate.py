#!/usr/bin/env python3
"""
Validate a dataset library (and optionally fix common issues).

This focuses on: "Is the OHLCV file engine-ready and sane?"
It does NOT try to be perfect market-data QA.

Checks (per dataset):
- required columns present (or aliases resolvable)
- ts parseable, monotonic increasing
- no duplicate ts
- dt parseable (or derivable from ts)
- o/h/l/c numeric and non-null
- basic range checks (high >= low, etc.)
- for daily interval: gap/missing-day estimate (warning)

Optional:
- --fix: sort + dedupe + recompute dt + rewrite CSV
- --write-catalog: update (or create) catalog.json with computed meta fields
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


CANON = ["ts", "dt", "open", "high", "low", "close", "volume"]
ALIASES = {
    "ts": ["ts", "timestamp", "open_time", "time", "t"],
    "dt": ["dt", "datetime", "date"],
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "c"],
    "volume": ["volume", "vol", "v"],
}


def repo_root() -> Path:
    p = Path(__file__).resolve()
    if len(p.parents) >= 2:
        return p.parents[1]
    return Path.cwd().resolve()


def _find_col(df: pd.DataFrame, key: str) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in ALIASES.get(key, [key]):
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def load_df(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Map/derive columns
    ts_col = _find_col(out, "ts")
    dt_col = _find_col(out, "dt")
    if ts_col is None and dt_col is None:
        raise ValueError("Missing both ts and dt.")

    # Numeric columns
    def get_num(key: str, required: bool = True) -> Optional[pd.Series]:
        col = _find_col(out, key)
        if col is None:
            if required:
                raise ValueError(f"Missing required column: {key}")
            return None
        return pd.to_numeric(out[col], errors="coerce")

    o = get_num("open", required=True)
    h = get_num("high", required=True)
    l = get_num("low", required=True)
    c = get_num("close", required=True)
    v = get_num("volume", required=False)
    if v is None:
        v = pd.Series([0.0] * len(out))

    if ts_col is None:
        dt = pd.to_datetime(out[dt_col], utc=True, errors="coerce")
        if dt.isna().all():
            raise ValueError("dt exists but could not be parsed to derive ts.")
        ts = (dt.view("int64") // 1_000_000).astype("int64")
    else:
        ts = pd.to_numeric(out[ts_col], errors="coerce")

    if dt_col is None:
        dt = pd.to_datetime(ts, unit="ms", utc=True, errors="coerce")
    else:
        dt = pd.to_datetime(out[dt_col], utc=True, errors="coerce")
        # If dt parsing fails but ts works, fall back.
        if dt.isna().all() and ts.notna().any():
            dt = pd.to_datetime(ts, unit="ms", utc=True, errors="coerce")

    canon = pd.DataFrame(
        {
            "ts": ts.astype("Int64"),
            "dt": dt.astype(str),
            "open": o.astype(float),
            "high": h.astype(float),
            "low": l.astype(float),
            "close": c.astype(float),
            "volume": v.astype(float),
        }
    )
    canon = canon.dropna(subset=["ts"])
    canon = canon.sort_values("ts").reset_index(drop=True)
    canon = canon.drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    return canon


def compute_daily_gaps(df: pd.DataFrame) -> Tuple[float, int, int]:
    """
    Returns (missing_days_pct, missing_days, expected_days).
    For non-daily data, this is just a rough heuristic.
    """
    if df.empty:
        return (0.0, 0, 0)
    dts = pd.to_datetime(df["dt"], utc=True, errors="coerce")
    dates = dts.dt.floor("D").dropna().unique()
    if len(dates) == 0:
        return (0.0, 0, 0)
    min_d = pd.to_datetime(min(dates)).to_pydatetime().date()
    max_d = pd.to_datetime(max(dates)).to_pydatetime().date()
    expected = (max_d - min_d).days + 1
    missing = max(0, expected - len(dates))
    pct = (missing / expected * 100.0) if expected > 0 else 0.0
    return (pct, missing, expected)


def validate_df(df: pd.DataFrame) -> List[str]:
    errs: List[str] = []
    if df.empty:
        errs.append("empty dataset")
        return errs

    # ts monotonic
    ts = df["ts"].astype("int64")
    if not ts.is_monotonic_increasing:
        errs.append("ts is not monotonic increasing (after canonicalize this should not happen)")

    # duplicate ts
    dup = ts.duplicated().sum()
    if dup > 0:
        errs.append(f"duplicate ts rows: {dup}")

    # dt parseable
    dt = pd.to_datetime(df["dt"], utc=True, errors="coerce")
    if dt.isna().any():
        errs.append(f"dt has {int(dt.isna().sum())} unparsable rows")

    # OHLC sanity
    if (df["high"] < df["low"]).any():
        errs.append("found rows where high < low")
    if (df[["open", "high", "low", "close"]] < 0).any().any():
        errs.append("found negative OHLC values")

    # Missing OHLC NaNs
    if df[["open", "high", "low", "close"]].isna().any().any():
        n = int(df[["open", "high", "low", "close"]].isna().sum().sum())
        errs.append(f"found {n} NaNs in OHLC")

    return errs


def load_catalog(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "datasets" in raw:
        raw = raw["datasets"]
    if not isinstance(raw, list):
        raise ValueError("catalog must be a list or {datasets:[...]}")
    return raw


def discover_paths(data_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for suf in ("*.csv", "*.parquet", "*.pq"):
        paths.extend(sorted(data_dir.rglob(suf)))
    return [p for p in paths if p.name.lower() != "catalog.json"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a dataset library directory or catalog")
    ap.add_argument("--catalog", help="Path to catalog.json (optional)")
    ap.add_argument("--data-dir", help="Directory to scan if no catalog is given")
    ap.add_argument("--fix", action="store_true", help="Attempt to fix common issues and rewrite CSVs")
    ap.add_argument("--write-catalog", help="Write an updated catalog JSON with computed meta fields")
    ap.add_argument("--interval", default="1d", help="Interval label (used for gap heuristic labels)")
    args = ap.parse_args()

    rr = repo_root()

    entries: List[Dict[str, Any]] = []
    dataset_paths: List[Tuple[Optional[Dict[str, Any]], Path]] = []

    if args.catalog:
        cat_path = Path(args.catalog).expanduser().resolve()
        entries = load_catalog(cat_path)
        for e in entries:
            fp = e.get("file_path") or e.get("path")
            if not fp:
                continue
            p = Path(fp)
            if not p.is_absolute():
                p = (rr / p).resolve()
            dataset_paths.append((e, p))
    else:
        if not args.data_dir:
            raise SystemExit("Provide --catalog or --data-dir")
        data_dir = Path(args.data_dir).expanduser().resolve()
        for p in discover_paths(data_dir):
            dataset_paths.append((None, p))

    if not dataset_paths:
        print("[validate] No dataset files found.")
        if args.write_catalog:
            Path(args.write_catalog).write_text("[]", encoding="utf-8")
        return 0

    errors_total = 0
    warnings_total = 0
    out_entries: List[Dict[str, Any]] = []

    for maybe_entry, path in dataset_paths:
        tag = (maybe_entry.get("symbol") if maybe_entry else None) or path.stem
        print(f"\n[{tag}] {path}")

        try:
            raw = load_df(path)
            canon = canonicalize(raw)
        except Exception as e:
            print(f"  ERROR: failed to load/canonicalize: {e}")
            errors_total += 1
            continue

        errs = validate_df(canon)
        for e in errs:
            print("  ERROR:", e)
        if errs:
            errors_total += 1

        # Daily gap estimate (warning)
        miss_pct, miss_days, expected = compute_daily_gaps(canon)
        if expected > 0 and miss_pct > 1.0:
            print(f"  WARN: missing-day estimate: {miss_pct:.2f}% ({miss_days}/{expected})")
            warnings_total += 1
        else:
            print(f"  OK: missing-day estimate: {miss_pct:.2f}% ({miss_days}/{expected})")

        # Meta summary
        start_dt = str(canon["dt"].iloc[0]) if len(canon) else None
        end_dt = str(canon["dt"].iloc[-1]) if len(canon) else None
        print(f"  Rows: {len(canon):,}  Range: {start_dt} -> {end_dt}")

        # Optional fix: rewrite in-place (CSV or Parquet)
        if args.fix:
            suf = path.suffix.lower()
            try:
                if suf == ".csv":
                    canon.to_csv(path, index=False)
                    print("  FIXED: rewrote canonicalized CSV (sorted, deduped, dt recomputed)")
                elif suf in (".parquet", ".pq"):
                    canon.to_parquet(path, index=False)
                    print("  FIXED: rewrote canonicalized Parquet (sorted, deduped, dt recomputed)")
            except Exception as e:
                print(f"  ERROR: failed to rewrite dataset: {e}")
                errors_total += 1

        # Build updated entry row (if requested)
        if args.write_catalog:
            if maybe_entry:
                e = dict(maybe_entry)
            else:
                # Minimal entry inferred from filename
                sym = path.stem.split(".")[0].split("_")[0].upper()
                rel = str(path.resolve())
                try:
                    rel = str(path.resolve().relative_to(rr))
                except Exception:
                    pass
                e = {
                    "id": f"spot_{args.interval}_binance_{sym}usdt".lower(),
                    "symbol": sym,
                    "name": sym,
                    "timeframe": "1D" if args.interval.lower() == "1d" else args.interval,
                    "source": "binance",
                    "quote": "USDT",
                    "pair": f"{sym}/USDT",
                    "file_path": rel,
                }

            e["rows"] = int(len(canon))
            e["start_dt"] = start_dt
            e["end_dt"] = end_dt
            e["quality"] = {
                "missing_days_pct": float(round(miss_pct, 4)),
                "missing_days": int(miss_days),
                "expected_days": int(expected),
            }
            out_entries.append(e)

    if args.write_catalog:
        out_path = Path(args.write_catalog).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # sort stable
        out_entries.sort(key=lambda d: d.get("symbol", ""))
        out_path.write_text(json.dumps(out_entries, indent=2), encoding="utf-8")
        print(f"\n[validate] Wrote updated catalog -> {out_path}")

    print(f"\n[validate] Done. errors={errors_total} warnings={warnings_total}")
    return 2 if errors_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
