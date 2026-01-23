# engine/walkforward.py
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.backtester import BacktestConfig, _efficiency_stats, _trade_stats, run_backtest_once
from engine.batch import (
    ConfigError,
    _ensure_vol_bps,
    _import_symbol,
    _instantiate_template,
    _load_ohlcv,
    _make_constraints,
    _normalize_columns,
    parse_strategy_config,
)
from engine.contracts import EngineConstraints, StrategyConfig
from engine.features import add_features

# ============================================================
# Worker globals (Windows spawn-safe)
# ============================================================

_WF_DF_FEAT: Optional[pd.DataFrame] = None
_WF_WINDOWS: Optional[List[Dict[str, Any]]] = None
_WF_TEMPLATE_CLS: Any = None
_WF_CONSTRAINTS: Optional[EngineConstraints] = None
_WF_ENGINE_CFG: Optional[BacktestConfig] = None


# ============================================================
# Small helpers
# ============================================================


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                obj = json.loads(s)
            except Exception as e:
                raise ConfigError(f"Line {line_no}: invalid JSONL in {path}") from e
            if not isinstance(obj, dict):
                raise ConfigError(f"Line {line_no}: JSONL row must be object in {path}")
            yield obj


def _chunked(xs: List[Any], n: int) -> List[List[Any]]:
    n = int(max(1, n))
    return [xs[i : i + n] for i in range(0, len(xs), n)]


def _infer_sort_defaults(meta: Dict[str, Any]) -> Tuple[str, bool]:
    sort_by = str(meta.get("final_sort_by") or "performance.sharpe")
    sort_desc = bool(meta.get("final_sort_desc", True))
    return sort_by, sort_desc


def _pick_top_config_ids(
    run_dir: Path,
    *,
    top_n: int,
    sort_by: str,
    sort_desc: bool,
) -> List[str]:
    p_full_pass = run_dir / "results_full_passed.csv"
    p_full = run_dir / "results_full.csv"

    src = p_full_pass if p_full_pass.exists() else p_full
    if not src.exists():
        raise FileNotFoundError(f"Missing results file: {src}")

    df = pd.read_csv(src)
    if df.empty:
        return []

    if "gates.passed" in df.columns:
        df = df[df["gates.passed"].astype(bool)].copy()

    if sort_by not in df.columns:
        sort_by = "equity.total_return" if "equity.total_return" in df.columns else df.columns[0]

    df[sort_by] = pd.to_numeric(df[sort_by], errors="coerce")
    df = df.dropna(subset=[sort_by])

    df = df.sort_values(sort_by, ascending=(not sort_desc)).head(int(top_n))

    if "config.id" not in df.columns:
        raise ValueError(f"results file missing 'config.id': {src}")

    return [str(x) for x in df["config.id"].tolist()]


def _load_configs_from_resolved(
    run_dir: Path, config_ids: List[str]
) -> List[Tuple[str, int, StrategyConfig, Dict[str, Any]]]:
    """
    Returns list of: (config_id, line_no, parsed StrategyConfig, normalized dict)
    """
    resolved = run_dir / "configs_resolved.jsonl"
    if not resolved.exists():
        raise FileNotFoundError(f"Missing configs_resolved.jsonl: {resolved}")

    want = set(config_ids)
    out: List[Tuple[str, int, StrategyConfig, Dict[str, Any]]] = []

    for row in _read_jsonl(resolved):
        cid = str(row.get("config_id", ""))
        if cid not in want:
            continue
        line_no = int(row.get("line_no", 0) or 0)
        normalized = row.get("normalized")
        if not isinstance(normalized, dict):
            continue

        cid2, cfg, _norm2 = parse_strategy_config(normalized, line_no=line_no)
        if cid2 != cid:
            raise ValueError(
                f"Config ID mismatch for {cid}: resolved hashes to {cid2}. "
                "This usually means hashing/normalization changed between runs."
            )

        out.append((cid, line_no, cfg, normalized))

    # Preserve requested order
    id_to = {cid: x for (cid, *_rest) in out for x in [x for x in out if x[0] == cid]}
    ordered: List[Tuple[str, int, StrategyConfig, Dict[str, Any]]] = []
    for cid in config_ids:
        found = [x for x in out if x[0] == cid]
        if found:
            ordered.append(found[0])

    return ordered


def _build_windows_by_ts(
    df_feat: pd.DataFrame,
    *,
    window_days: int,
    step_days: int,
    min_bars: int = 1000,
) -> List[Dict[str, Any]]:
    if "ts" not in df_feat.columns or "dt" not in df_feat.columns:
        raise ValueError("df_feat must include 'ts' and 'dt' columns")

    ts = pd.to_numeric(df_feat["ts"], errors="coerce").dropna().astype(np.int64).to_numpy()
    if len(ts) < min_bars:
        return []

    ts_sorted = ts  # df is already sorted in pipeline

    window_ms = int(window_days) * 86_400_000
    step_ms = int(step_days) * 86_400_000

    t0 = int(ts_sorted[0])
    tN = int(ts_sorted[-1])

    out: List[Dict[str, Any]] = []
    k = 0
    start = t0
    while start + window_ms <= tN:
        i0 = int(np.searchsorted(ts_sorted, start, side="left"))
        i1 = int(np.searchsorted(ts_sorted, start + window_ms, side="left"))
        bars = i1 - i0
        if bars >= int(min_bars):
            out.append(
                {
                    "window_idx": int(k),
                    "start_i": int(i0),
                    "end_i": int(i1),
                    "start_ts": int(ts_sorted[i0]),
                    "end_ts_excl": int(start + window_ms),
                    "start_dt": str(df_feat["dt"].iloc[i0]),
                    "end_dt": str(df_feat["dt"].iloc[i1 - 1]),
                    "bars": int(bars),
                }
            )
            k += 1
        start += step_ms

    return out


# ============================================================
# Worker
# ============================================================


def _wf_worker_init(
    df_feat_path: str,
    template_dotted: str,
    windows: List[Dict[str, Any]],
) -> None:
    global _WF_DF_FEAT, _WF_WINDOWS, _WF_TEMPLATE_CLS, _WF_CONSTRAINTS, _WF_ENGINE_CFG

    _WF_DF_FEAT = pd.read_parquet(df_feat_path)
    _WF_WINDOWS = list(windows)
    _WF_TEMPLATE_CLS = _import_symbol(template_dotted)
    _WF_CONSTRAINTS = _make_constraints()
    _WF_ENGINE_CFG = BacktestConfig()


def _run_one_window(
    df_window: pd.DataFrame,
    cfg: StrategyConfig,
    *,
    seed: int,
    starting_equity: float,
    constraints: EngineConstraints,
    engine_cfg: BacktestConfig,
) -> Dict[str, Any]:
    """
    Output-only: no equity curve, no fills.
    Returns cheap stats (return + trades).
    """
    strategy = _instantiate_template(_WF_TEMPLATE_CLS, cfg)

    metrics, _fills_df, _eq_df, trades_df, _guard = run_backtest_once(
        df=df_window,
        strategy=strategy,
        seed=int(seed),
        starting_equity=float(starting_equity),
        constraints=constraints,
        cfg=engine_cfg,
        show_progress=False,
        features_ready=True,
        record_fills=False,
        record_equity_curve=False,
    )

    tstats = _trade_stats(trades_df)
    efficiency = _efficiency_stats(trades_df)

    eq = metrics.get("equity", {})
    return {
        "equity": eq,
        "trades_summary": tstats,
        "efficiency": efficiency,
    }


def _wf_run_config_chunk(
    chunk: List[Tuple[str, StrategyConfig, str, str]],
    seed: int,
    starting_equity: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (wf_rows, wf_summaries) for this chunk.
    """
    if _WF_DF_FEAT is None or _WF_WINDOWS is None or _WF_CONSTRAINTS is None or _WF_ENGINE_CFG is None:
        raise RuntimeError("WF worker not initialized")

    df_feat = _WF_DF_FEAT
    windows = _WF_WINDOWS
    constraints = _WF_CONSTRAINTS
    engine_cfg = _WF_ENGINE_CFG

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for config_id, cfg, strategy_name, side in chunk:
        returns: List[float] = []
        trades: List[int] = []

        for w in windows:
            i0 = int(w["start_i"])
            i1 = int(w["end_i"])
            df_w = df_feat.iloc[i0:i1]

            res = _run_one_window(
                df_window=df_w,
                cfg=cfg,
                seed=int(seed),
                starting_equity=float(starting_equity),
                constraints=constraints,
                engine_cfg=engine_cfg,
            )

            eq = res["equity"]
            tstats = res["trades_summary"]

            r = float(eq.get("total_return", 0.0) or 0.0)
            ntr = int(tstats.get("trades_closed", 0) or 0)

            returns.append(r)
            trades.append(ntr)

            all_rows.append(
                {
                    "config_id": str(config_id),
                    "strategy_name": str(strategy_name),
                    "side": str(side),
                    "window_idx": int(w["window_idx"]),
                    "window_start_dt": str(w["start_dt"]),
                    "window_end_dt": str(w["end_dt"]),
                    "bars": int(w["bars"]),
                    "equity.total_return": float(r),
                    "equity.start": float(eq.get("start", starting_equity) or starting_equity),
                    "equity.end": float(eq.get("end", starting_equity) or starting_equity),
                    "trades_summary.trades_closed": int(ntr),
                    "trades_summary.win_rate": float(tstats.get("win_rate", 0.0) or 0.0),
                    "trades_summary.profit_factor": float(
                        tstats.get("profit_factor", 0.0) or 0.0
                    ),
                    "trades_summary.expectancy": float(
                        tstats.get("expectancy", 0.0) or 0.0
                    ),
                    "efficiency.fee_impact_pct": float(
                        res["efficiency"].get("fee_impact_pct", 0.0) or 0.0
                    ),
                }
            )

        # Summary across windows (output-only, no gating yet)
        arr = np.array(returns, dtype=float)
        tr = np.array(trades, dtype=float)

        n = int(len(arr))
        if n == 0:
            continue

        summaries.append(
            {
                "config_id": str(config_id),
                "strategy_name": str(strategy_name),
                "side": str(side),
                "windows": int(n),
                "pct_profitable_windows": float((arr > 0.0).mean()) if n else 0.0,
                "mean_window_return": float(arr.mean()) if n else 0.0,
                "median_window_return": float(np.median(arr)) if n else 0.0,
                "min_window_return": float(arr.min()) if n else 0.0,
                "max_window_return": float(arr.max()) if n else 0.0,
                "mean_trades_per_window": float(tr.mean()) if n else 0.0,
                "min_trades_per_window": float(tr.min()) if n else 0.0,
                "max_trades_per_window": float(tr.max()) if n else 0.0,
            }
        )

    return all_rows, summaries


# ============================================================
# CLI
# ============================================================


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Walk-forward runner (rolling windows)")
    ap.add_argument(
        "--from-run",
        required=True,
        help="Path to batch run folder (contains results_full.csv + configs_resolved.jsonl)",
    )
    ap.add_argument(
        "--data",
        default=None,
        help="Override data path. If omitted, uses batch_meta.json['data']",
    )
    ap.add_argument("--template", default=None, help="Override template class path")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--step-days", type=int, default=15)
    ap.add_argument("--min-bars", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--starting-equity", type=float, default=None)
    ap.add_argument("--sort-by", default=None)
    ap.add_argument(
        "--sort-desc",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Sort descending (default from batch_meta.json if present).",
    )

    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--chunk-size", type=int, default=5)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--out", default=None, help="Output folder (default: under from-run)")
    args = ap.parse_args(argv)

    run_dir = Path(args.from_run).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"from-run not found: {run_dir}")

    meta_path = run_dir / "batch_meta.json"
    meta: Dict[str, Any] = _read_json(meta_path) if meta_path.exists() else {}

    data_path = str(args.data) if args.data else str(meta.get("data"))
    if not data_path:
        raise ValueError("No data path. Provide --data or ensure batch_meta.json has 'data'.")

    template = str(args.template) if args.template else str(meta.get("template") or "strategies.universal:UniversalStrategy")
    seed = int(args.seed) if args.seed is not None else int(meta.get("seed", 1))
    starting_equity = float(args.starting_equity) if args.starting_equity is not None else float(meta.get("starting_equity", 1000.0))

    sort_by_default, sort_desc_default = _infer_sort_defaults(meta)
    sort_by = str(args.sort_by) if args.sort_by else sort_by_default
    sort_desc = bool(args.sort_desc) if args.sort_desc is not None else bool(sort_desc_default)

    # Output folder
    tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out).resolve() if args.out else (run_dir / f"walkforward_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load df_feat (reuse cached parquet if possible)
    df_feat_path = run_dir / "df_feat.parquet"
    if df_feat_path.exists() and args.data is None:
        t0 = time.time()
        df_feat = pd.read_parquet(df_feat_path)
        load_sec = time.time() - t0
        print(f"\nLoaded df_feat from run: {df_feat_path} ({load_sec:.2f}s)")
    else:
        df = _load_ohlcv(data_path)
        df = _normalize_columns(df)
        df = _ensure_vol_bps(df, window=int(meta.get("vol_window", 60)))
        if "liq_mult" not in df.columns:
            df["liq_mult"] = 1.0
        else:
            df["liq_mult"] = pd.to_numeric(df["liq_mult"], errors="coerce").fillna(1.0)

        t0 = time.time()
        df_feat = add_features(df)
        feat_sec = time.time() - t0
        df_feat_path = out_dir / "df_feat.parquet"
        df_feat.to_parquet(df_feat_path, index=False)
        print(f"\nComputed df_feat: {df_feat_path} ({feat_sec:.2f}s)")

    # Windows
    windows = _build_windows_by_ts(
        df_feat,
        window_days=int(args.window_days),
        step_days=int(args.step_days),
        min_bars=int(args.min_bars),
    )
    if not windows:
        try:
            ts = pd.to_numeric(df_feat["ts"], errors="coerce").dropna().astype(np.int64)
            if len(ts) >= 2:
                span_days = (int(ts.iloc[-1]) - int(ts.iloc[0])) / 86_400_000.0
                raise ValueError(
                    "No windows built. "
                    f"Data span is ~{span_days:.1f} days; "
                    f"requested window_days={int(args.window_days)}. "
                    "Try reducing --window-days / --min-bars or pass --data to a longer dataset."
                )
        except Exception:
            pass
        raise ValueError("No windows built. Try reducing --min-bars or window size.")

    print(
        f"\nWindows: {len(windows)}  "
        f"(window_days={args.window_days}, step_days={args.step_days}, min_bars={args.min_bars})"
    )
    print(f"First window: {windows[0]['start_dt']} -> {windows[0]['end_dt']} ({windows[0]['bars']} bars)")
    print(f"Last window:  {windows[-1]['start_dt']} -> {windows[-1]['end_dt']} ({windows[-1]['bars']} bars)")

    # Select configs
    config_ids = _pick_top_config_ids(
        run_dir,
        top_n=int(args.top_n),
        sort_by=sort_by,
        sort_desc=sort_desc,
    )
    if not config_ids:
        raise ValueError("No config IDs selected from results_full*.csv")

    cfgs = _load_configs_from_resolved(run_dir, config_ids)

    # Persist configs used
    (out_dir / "config_ids.txt").write_text("\n".join(config_ids) + "\n", encoding="utf-8")

    wf_meta = {
        "from_run": str(run_dir),
        "data": str(data_path),
        "template": str(template),
        "seed": int(seed),
        "starting_equity": float(starting_equity),
        "top_n": int(args.top_n),
        "selected_configs": int(len(cfgs)),
        "sort_by": str(sort_by),
        "sort_desc": bool(sort_desc),
        "window_days": int(args.window_days),
        "step_days": int(args.step_days),
        "min_bars": int(args.min_bars),
        "windows": int(len(windows)),
        "jobs": int(args.jobs),
        "chunk_size": int(args.chunk_size),
        "df_feat_path": str(df_feat_path),
        "out_dir": str(out_dir),
    }
    _write_json(out_dir / "wf_meta.json", wf_meta)

    # Build tasks
    tasks: List[Tuple[str, StrategyConfig, str, str]] = [
        (cid, cfg, cfg.strategy_name, cfg.side) for (cid, _ln, cfg, _norm) in cfgs
    ]
    chunks = _chunked(tasks, int(max(1, args.chunk_size)))

    # Run
    jobs = int(max(1, args.jobs))
    t_run0 = time.time()

    wf_rows: List[Dict[str, Any]] = []
    wf_summaries: List[Dict[str, Any]] = []

    use_tqdm = False
    pbar = None
    if not args.no_progress:
        try:
            from tqdm import tqdm

            pbar = tqdm(total=len(tasks), desc="WF", unit="cfg")
            use_tqdm = True
        except Exception:
            use_tqdm = False

    if jobs == 1:
        _wf_worker_init(str(df_feat_path), str(template), windows)
        for ch in chunks:
            rows, sums = _wf_run_config_chunk(
                ch,
                seed=int(seed),
                starting_equity=float(starting_equity),
            )
            wf_rows.extend(rows)
            wf_summaries.extend(sums)
            if use_tqdm and pbar is not None:
                pbar.update(len(ch))
    else:
        with ProcessPoolExecutor(
            max_workers=jobs,
            initializer=_wf_worker_init,
            initargs=(str(df_feat_path), str(template), windows),
        ) as ex:
            fut_to_n: Dict[Any, int] = {}
            futs = []
            for ch in chunks:
                fut = ex.submit(
                    _wf_run_config_chunk,
                    ch,
                    int(seed),
                    float(starting_equity),
                )
                futs.append(fut)
                fut_to_n[fut] = int(len(ch))
            for fut in as_completed(futs):
                rows, sums = fut.result()
                wf_rows.extend(rows)
                wf_summaries.extend(sums)
                if use_tqdm and pbar is not None:

                    pbar.update(fut_to_n.get(fut, 1))

    if use_tqdm and pbar is not None:
        pbar.close()

    elapsed = time.time() - t_run0

    # Save
    wf_rows_df = pd.DataFrame(wf_rows)
    wf_sum_df = pd.DataFrame(wf_summaries)

    wf_rows_df.to_csv(out_dir / "wf_results.csv", index=False)
    wf_sum_df.to_csv(out_dir / "wf_summary.csv", index=False)

    print(f"\nWalk-forward complete. Output: {out_dir}")
    print(f"Configs: {len(tasks)}   Windows: {len(windows)}")
    print(f"Rows:    {len(wf_rows_df)} (config x window)")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Files:   wf_results.csv, wf_summary.csv, wf_meta.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())