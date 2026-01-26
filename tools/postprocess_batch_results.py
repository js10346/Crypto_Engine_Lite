from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def pareto_front(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    maximize_x: bool = True,
    maximize_y: bool = True,
) -> pd.Series:
    """
    Returns boolean mask for Pareto-efficient points.
    O(n^2) is fine for ~200-2000 rows; keep it simple.
    """
    x = _to_num(df[x_col]).to_numpy(dtype=float)
    y = _to_num(df[y_col]).to_numpy(dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    idxs = np.where(ok)[0]
    eff = np.zeros(len(df), dtype=bool)

    for i in idxs:
        dominated = False
        for j in idxs:
            if i == j:
                continue
            xi, yi = x[i], y[i]
            xj, yj = x[j], y[j]

            better_x = (xj >= xi) if maximize_x else (xj <= xi)
            better_y = (yj >= yi) if maximize_y else (yj <= yi)
            strictly_better = ((xj > xi) if maximize_x else (xj < xi)) or (
                (yj > yi) if maximize_y else (yj < yi)
            )

            if better_x and better_y and strictly_better:
                dominated = True
                break
        eff[i] = not dominated

    return pd.Series(eff, index=df.index)


def main() -> int:
    ap = argparse.ArgumentParser(description="Postprocess batch results_full.csv for DCA lab")
    ap.add_argument("--from-run", required=True, help="Path to runs/batch_... folder")
    ap.add_argument("--out", default=None, help="Output folder (default: from-run/post)")
    ap.add_argument("--top-n", type=int, default=50)

    # Filters
    ap.add_argument("--max-dd", type=float, default=None, help="Max equity drawdown (0-1)")
    ap.add_argument("--min-profit", type=float, default=None, help="Min net profit ex deposits")
    ap.add_argument("--min-twr", type=float, default=None, help="Min TWR total return (0-...)")

    # Scoring
    ap.add_argument(
        "--score",
        default="calmar_equity",
        choices=["calmar_equity", "profit_dd", "twr_dd", "profit"],
        help="Ranking score definition",
    )
    ap.add_argument(
        "--dd-penalty",
        type=float,
        default=10.0,
        help="Penalty multiplier for profit_dd score (higher penalizes DD more)",
    )

    args = ap.parse_args()
    run_dir = Path(args.from_run).resolve()
    src = run_dir / "results_full.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing results_full.csv: {src}")

    out_dir = Path(args.out).resolve() if args.out else (run_dir / "post")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    if df.empty:
        (out_dir / "ranked.csv").write_text("", encoding="utf-8")
        print(f"No rows in {src}")
        return 0

    # Required cols (these exist in your current pipeline)
    profit = _to_num(df.get("equity.net_profit_ex_cashflows", np.nan)).fillna(0.0)
    twr = _to_num(df.get("performance.twr_total_return", np.nan)).fillna(0.0)
    ann = _to_num(df.get("performance.annualized_return", np.nan)).fillna(0.0)
    dd_eq = _to_num(df.get("performance.max_drawdown_equity", np.nan)).fillna(0.0)

    eps = 1e-9
    df["score.profit"] = profit
    df["score.twr_dd"] = twr / (dd_eq + eps)
    df["score.calmar_equity"] = ann / (dd_eq + eps)
    df["score.profit_dd"] = profit / (1.0 + float(args.dd_penalty) * dd_eq)

    # Filters
    if args.max_dd is not None:
        df = df[dd_eq <= float(args.max_dd)].copy()
    if args.min_profit is not None:
        df = df[profit >= float(args.min_profit)].copy()
    if args.min_twr is not None:
        df = df[twr >= float(args.min_twr)].copy()

    if df.empty:
        (out_dir / "ranked.csv").write_text("", encoding="utf-8")
        print("No rows after filters.")
        return 0

    score_col = f"score.{str(args.score)}"
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.sort_values(score_col, ascending=False)

    # Pareto front flag: maximize profit, minimize drawdown
    df["pareto.profit_vs_dd"] = pareto_front(
        df,
        x_col="equity.net_profit_ex_cashflows",
        y_col="performance.max_drawdown_equity",
        maximize_x=True,
        maximize_y=False,
    ).astype(int)

    df.to_csv(out_dir / "ranked.csv", index=False)

    top_n = int(max(1, args.top_n))
    top = df.head(top_n).copy()
    top.to_csv(out_dir / "top.csv", index=False)

    # Convenience shortlist (ids)
    ids = top["config.id"].astype(str).tolist() if "config.id" in top.columns else []
    (out_dir / "top_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    print(f"Wrote: {out_dir / 'ranked.csv'}")
    print(f"Wrote: {out_dir / 'top.csv'}")
    print(f"Wrote: {out_dir / 'top_ids.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())