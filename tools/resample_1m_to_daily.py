from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def resample_1m_to_1d(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Require ts in ms (your pipeline uses ms)
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["ts"]).copy()
    df["ts"] = df["ts"].astype(np.int64)

    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.sort_values("ts").reset_index(drop=True)

    df["date"] = df["dt"].dt.floor("D")

    agg = df.groupby("date", as_index=False).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum") if "volume" in df.columns else ("close", "size"),
    )

    # ts = daily open in ms
    agg["ts"] = (pd.to_datetime(agg["date"], utc=True).astype("int64") // 1_000_000).astype(
        np.int64
    )
    agg["dt"] = pd.to_datetime(agg["ts"], unit="ms", utc=True).astype(str)

    # Column order expected by your pipeline
    out = agg[["ts", "dt", "open", "high", "low", "close", "volume"]].copy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="outp", required=True)
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.outp)
    outp.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(inp) if inp.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(inp)
    df1d = resample_1m_to_1d(df)
    df1d.to_csv(outp, index=False)
    print(f"Wrote: {outp}  rows={len(df1d)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())