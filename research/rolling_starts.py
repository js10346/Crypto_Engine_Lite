from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


import numpy as np
import pandas as pd

# --- Make repo root importable when running as a script (Windows-friendly) ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------------------------------------------

from engine.backtester import (  # noqa: E402
    BacktestConfig,
    _build_cashflow_performance_stats,
    run_backtest_once,
)
from engine.batch import (
    _ensure_vol_bps,
    _import_symbol,
    _instantiate_template,
    _load_ohlcv,
    _make_constraints,
    _normalize_columns,
)  # noqa: E402
from engine.contracts import StrategyConfig  # noqa: E402
from engine.features import add_features  # noqa: E402


# ============================================================
# Progress (optional): JSONL progress events for UI monitoring
# ============================================================

class ProgressWriter:
    """Append-only JSONL progress writer (safe no-op if path is None)."""

    def __init__(self, path: Optional[str]):
        self.enabled = bool(path)
        self.path: Optional[Path] = Path(path).resolve() if path else None
        self.t0 = 0.0
        self._fh = None
        if self.path:
            self.t0 = float(__import__("time").time())
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.path.write_text("", encoding="utf-8")
            except Exception:
                pass
            try:
                self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
            except Exception:
                self._fh = None
                self.enabled = False

    def write(self, obj: Dict[str, Any]) -> None:
        if not self.enabled or self._fh is None:
            return
        payload = dict(obj)
        t = float(__import__("time").time())
        payload.setdefault("t", t)
        payload.setdefault("t_rel", float(t - self.t0))
        try:
            self._fh.write(json.dumps(payload) + "\n")
            self._fh.flush()
        except Exception:
            pass

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(json.loads(s))
    return out


def _load_configs(
    run_dir: Path, config_ids: List[str]
) -> List[Tuple[str, StrategyConfig, Dict[str, Any]]]:
    resolved = run_dir / "configs_resolved.jsonl"
    if not resolved.exists():
        raise FileNotFoundError(f"Missing: {resolved}")

    want = set(config_ids)
    rows = _read_jsonl(resolved)

    out: List[Tuple[str, StrategyConfig, Dict[str, Any]]] = []
    for r in rows:
        cid = str(r.get("config_id", ""))
        if cid not in want:
            continue
        norm = r.get("normalized")
        if not isinstance(norm, dict):
            continue

        cfg = StrategyConfig(
            strategy_name=str(norm.get("strategy_name", "dca_swing")),
            side=str(norm.get("side", "long")),
            params=dict(norm.get("params") or {}),
        )
        out.append((cid, cfg, norm))

    id_to = {cid: (cid, cfg, norm) for cid, cfg, norm in out}
    return [id_to[cid] for cid in config_ids if cid in id_to]


def _slice_stats(values: np.ndarray) -> Dict[str, Any]:
    if values.size == 0:
        return {"n": 0}
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "p10": float(np.quantile(v, 0.10)),
        "p50": float(np.quantile(v, 0.50)),
        "p90": float(np.quantile(v, 0.90)),
        "mean": float(np.mean(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }

def _summarize_from_detail(df_detail: pd.DataFrame) -> pd.DataFrame:
    """
    Build rolling-start summary from detail rows (config_id x start).
    """
    if df_detail is None or df_detail.empty:
        return pd.DataFrame([])

    def q(series: pd.Series, p: float) -> float:
        s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
        if len(s) == 0:
            return float("nan")
        return float(np.quantile(s.to_numpy(dtype=float), p))

    out_rows: List[Dict[str, Any]] = []
    for cid, g in df_detail.groupby("config_id"):
        out_rows.append(
            {
                "config_id": str(cid),
                "windows": int(len(g)),
                "profit_p10": q(g["equity.net_profit_ex_cashflows"], 0.10)
                if "equity.net_profit_ex_cashflows" in g.columns
                else float("nan"),
                "profit_p50": q(g["equity.net_profit_ex_cashflows"], 0.50)
                if "equity.net_profit_ex_cashflows" in g.columns
                else float("nan"),
                "profit_p90": q(g["equity.net_profit_ex_cashflows"], 0.90)
                if "equity.net_profit_ex_cashflows" in g.columns
                else float("nan"),
                "twr_p10": q(g["performance.twr_total_return"], 0.10)
                if "performance.twr_total_return" in g.columns
                else float("nan"),
                "twr_p50": q(g["performance.twr_total_return"], 0.50)
                if "performance.twr_total_return" in g.columns
                else float("nan"),
                "twr_p90": q(g["performance.twr_total_return"], 0.90)
                if "performance.twr_total_return" in g.columns
                else float("nan"),
                "dd_p10": q(g["performance.max_drawdown_equity"], 0.10)
                if "performance.max_drawdown_equity" in g.columns
                else float("nan"),
                "dd_p50": q(g["performance.max_drawdown_equity"], 0.50)
                if "performance.max_drawdown_equity" in g.columns
                else float("nan"),
                "dd_p90": q(g["performance.max_drawdown_equity"], 0.90)
                if "performance.max_drawdown_equity" in g.columns
                else float("nan"),
                "uw_p50_days": q(g["uw_max_days"], 0.50)
                if "uw_max_days" in g.columns
                else float("nan"),
                "uw_p90_days": q(g["uw_max_days"], 0.90)
                if "uw_max_days" in g.columns
                else float("nan"),
                "util_p50": q(g["util_mean"], 0.50) if "util_mean" in g.columns else float("nan"),
                "util_p90": q(g["util_mean"], 0.90) if "util_mean" in g.columns else float("nan"),
            }
        )

    df_sum = pd.DataFrame(out_rows)
    # Robustness score (keep consistent with your previous definition)
    if "twr_p50" in df_sum.columns and "dd_p90" in df_sum.columns:
        df_sum["robustness_score"] = pd.to_numeric(df_sum["twr_p50"], errors="coerce").fillna(
            0.0
        ) - 0.5 * pd.to_numeric(df_sum["dd_p90"], errors="coerce").fillna(0.0)
    else:
        df_sum["robustness_score"] = 0.0

    return df_sum

def _max_underwater_days(eq: pd.Series) -> float:
    """
    Longest consecutive streak where equity is below its prior running peak.
    Daily bars => days == bars (1 bar = 1 day).
    """
    if eq is None or len(eq) == 0:
        return 0.0
    e = pd.to_numeric(eq, errors="coerce").fillna(0.0).astype(float).to_numpy()
    peak = -float("inf")
    cur = 0
    best = 0
    for x in e:
        if x > peak:
            peak = x
        if x < peak - 1e-12:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return float(best)  # 1 bar = 1 day in daily mode


def _util_mean(eq_df: pd.DataFrame) -> float:
    """
    Mean invested fraction over time:
      util_t = (pos_qty * price) / equity
    Clamped to [0,1] per bar; ignores bars with non-positive equity.
    """
    if eq_df is None or eq_df.empty:
        return 0.0
    if "pos_qty" not in eq_df.columns or "price" not in eq_df.columns or "equity" not in eq_df.columns:
        return 0.0
    pos = pd.to_numeric(eq_df["pos_qty"], errors="coerce").fillna(0.0).astype(float)
    px = pd.to_numeric(eq_df["price"], errors="coerce").fillna(0.0).astype(float)
    eq = pd.to_numeric(eq_df["equity"], errors="coerce").fillna(0.0).astype(float)
    invested = (pos.clip(lower=0.0) * px).astype(float)
    denom = eq.where(eq > 1e-9)
    util = (invested / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    util = util.clip(lower=0.0, upper=1.0)
    return float(util.mean()) if len(util) else 0.0


# ============================================================
# Parallelization: per-config multiprocessing (optional)
# ============================================================

_W_DF_FEAT = None
_W_TEMPLATE_CLS = None
_W_CONSTRAINTS = None
_W_ENGINE_CFG = None
_W_SEED = 1
_W_STARTING_EQUITY = 1000.0
_W_STARTS = None
_W_MIN_BARS = 365


def _worker_init(
    df_feat_parquet: str,
    template: str,
    starts: List[int],
    min_bars: int,
    seed: int,
    starting_equity: float,
) -> None:
    """Initializer for ProcessPool workers."""
    global _W_DF_FEAT, _W_TEMPLATE_CLS, _W_CONSTRAINTS, _W_ENGINE_CFG, _W_SEED, _W_STARTING_EQUITY, _W_STARTS, _W_MIN_BARS
    _W_DF_FEAT = pd.read_parquet(df_feat_parquet)
    _W_TEMPLATE_CLS = _import_symbol(str(template))
    _W_CONSTRAINTS = _make_constraints()
    _W_ENGINE_CFG = BacktestConfig(market_mode="spot")
    _W_SEED = int(seed)
    _W_STARTING_EQUITY = float(starting_equity)
    _W_STARTS = list(starts)
    _W_MIN_BARS = int(min_bars)


def _process_one_config(payload: Tuple[str, Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]], float]:
    """Run all rolling starts for one config. Returns (config_id, rows, median_twr)."""
    cid, cfg_payload = payload
    assert _W_DF_FEAT is not None
    assert _W_TEMPLATE_CLS is not None
    assert _W_CONSTRAINTS is not None
    assert _W_ENGINE_CFG is not None
    assert _W_STARTS is not None

    cfg = StrategyConfig(
        strategy_name=str(cfg_payload.get("strategy_name", "dca_swing")),
        side=str(cfg_payload.get("side", "long")),
        params=dict(cfg_payload.get("params") or {}),
    )

    rows_for_cfg: List[Dict[str, Any]] = []
    twrs: List[float] = []

    df_feat = _W_DF_FEAT
    for s in _W_STARTS:
        df_w = df_feat.iloc[int(s):]
        if len(df_w) < int(_W_MIN_BARS):
            continue

        strategy = _instantiate_template(_W_TEMPLATE_CLS, cfg)

        metrics, _fills, eq_df, _trades, _guard = run_backtest_once(
            df=df_w,
            strategy=strategy,
            seed=int(_W_SEED),
            starting_equity=float(_W_STARTING_EQUITY),
            constraints=_W_CONSTRAINTS,
            cfg=_W_ENGINE_CFG,
            show_progress=False,
            features_ready=True,
            record_fills=False,
            record_equity_curve=True,
        )

        eq = metrics.get("equity", {}) if isinstance(metrics, dict) else {}
        perf = _build_cashflow_performance_stats(df_w, eq_df)

        uw_days = (
            _max_underwater_days(eq_df["equity"])
            if (eq_df is not None and "equity" in eq_df.columns)
            else 0.0
        )
        util_mean = _util_mean(eq_df)

        twr = float(perf.get("twr_total_return", np.nan))
        twrs.append(twr)

        row = {
            "config_id": str(cid),
            "start_i": int(s),
            "start_dt": str(df_feat["dt"].iloc[int(s)]) if "dt" in df_feat.columns else "",
            "bars": int(len(df_w)),
            "equity.end": float(eq.get("end", np.nan)),
            "equity.cashflow_total": float(eq.get("cashflow_total", np.nan)),
            "equity.net_profit_ex_cashflows": float(eq.get("net_profit_ex_cashflows", np.nan)),
            "performance.twr_total_return": twr,
            "performance.annualized_return": float(perf.get("annualized_return", np.nan)),
            "performance.max_drawdown_equity": float(perf.get("max_drawdown_equity", np.nan)),
            "uw_max_days": float(uw_days),
            "util_mean": float(util_mean),
        }
        rows_for_cfg.append(row)

    try:
        _vals = np.array([float(x) for x in twrs], dtype=float)
        cfg_score = float(np.nanmedian(_vals)) if _vals.size else float("nan")
    except Exception:
        cfg_score = float("nan")

    return str(cid), rows_for_cfg, float(cfg_score)


def main() -> int:
    ap = argparse.ArgumentParser(description="Rolling-start sensitivity (bar-step starts)")
    ap.add_argument("--from-run", required=True, help="Batch run folder: runs/batch_...")
    ap.add_argument(
        "--template",
        default="strategies.dca_swing:Strategy",
        help="Strategy template used in that batch run",
    )
    ap.add_argument(
        "--ids",
        default=None,
        help="Optional path to ids file (default: from-run/post/top_ids.txt)",
    )
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=1, help="Number of worker processes (parallel configs).")
    ap.add_argument("--start-step", type=int, default=30, help="Bars between start points")
    ap.add_argument("--min-bars", type=int, default=365, help="Min bars per run")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--starting-equity", type=float, default=1000.0)
    ap.add_argument("--out", default=None, help="Output folder (default: from-run/rolling_starts)")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--progress-file", default=None, help="Optional JSONL progress file for UI monitoring.")
    ap.add_argument("--progress-every", type=int, default=1, help="Emit progress every N configs.")
    args = ap.parse_args()

    run_dir = Path(args.from_run).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"from-run not found: {run_dir}")

    ids_path = Path(args.ids).resolve() if args.ids else (run_dir / "post" / "top_ids.txt")
    if not ids_path.exists():
        raise FileNotFoundError(f"Missing ids file: {ids_path}")

    config_ids = [
        x.strip()
        for x in ids_path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ][: int(args.top_n)]
    if not config_ids:
        raise ValueError("No config IDs provided")

    df_feat_path = run_dir / "df_feat.parquet"
    if df_feat_path.exists():
        df_feat = pd.read_parquet(df_feat_path)
    else:
        # Fallback: rebuild df_feat from batch_meta.json data path
        meta_path = run_dir / "batch_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Missing df_feat.parquet and batch_meta.json: {df_feat_path} / {meta_path}"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        data_path = str(meta.get("data") or "")
        if not data_path:
            raise ValueError(
                "batch_meta.json missing 'data'. Provide a run folder created by engine.batch "
                "or rerun batch with --jobs > 1 to emit df_feat.parquet."
            )

        vol_window = int(meta.get("vol_window", 60))
        features_ready = bool(meta.get("features_ready", False))

        df = _load_ohlcv(data_path)
        df = _normalize_columns(df)
        df = _ensure_vol_bps(df, window=vol_window)
        if "liq_mult" not in df.columns:
            df["liq_mult"] = 1.0
        else:
            df["liq_mult"] = pd.to_numeric(df["liq_mult"], errors="coerce").fillna(1.0)

        df_feat = df if features_ready else add_features(df)

        # Cache for next runs (even if batch was jobs=1)
        try:
            df_feat.to_parquet(df_feat_path, index=False)
        except Exception:
            pass

    cfgs = _load_configs(run_dir, config_ids)

    template_cls = _import_symbol(str(args.template))
    constraints = _make_constraints()
    engine_cfg = BacktestConfig(market_mode="spot")

    out_dir = Path(args.out).resolve() if args.out else (run_dir / "rolling_starts")
    out_dir.mkdir(parents=True, exist_ok=True)

    progress = ProgressWriter(args.progress_file)
    progress.write({"stage": "rolling_starts", "phase": "init", "done": 0, "total": int(len(config_ids)), "out_dir": str(out_dir)})
    # Accumulation: load prior results if they exist and parameters match
    meta_path = out_dir / "rs_meta.json"
    detail_path = out_dir / "rolling_starts_detail.csv"
    summary_path = out_dir / "rolling_starts_summary.csv"

    start_step = int(max(1, args.start_step))
    min_bars = int(max(1, args.min_bars))

    meta_now = {
        "template": str(args.template),
        "seed": int(args.seed),
        "starting_equity": float(args.starting_equity),
        "start_step": int(start_step),
        "min_bars": int(min_bars),
        "note": "rolling-start stats are only comparable when these parameters match",
    }

    accumulate_ok = True
    if meta_path.exists():
        try:
            meta_prev = json.loads(meta_path.read_text(encoding="utf-8"))
            # Only allow accumulation if key params match
            keys = ["template", "seed", "starting_equity", "start_step", "min_bars"]
            for k in keys:
                if meta_prev.get(k) != meta_now.get(k):
                    accumulate_ok = False
                    break
        except Exception:
            accumulate_ok = False

    df_prev_detail = pd.DataFrame([])
    if accumulate_ok and detail_path.exists():
        try:
            df_prev_detail = pd.read_csv(detail_path)
        except Exception:
            df_prev_detail = pd.DataFrame([])

    already_done: set[str] = set()
    if accumulate_ok and (not df_prev_detail.empty) and "config_id" in df_prev_detail.columns:
        already_done = set(df_prev_detail["config_id"].astype(str).unique().tolist())
    elif (not accumulate_ok) and (detail_path.exists() or summary_path.exists()):
        print(
            "Warning: rolling-start parameters changed; overwriting existing rolling-start outputs."
        )

    n = int(len(df_feat))
    starts = [i for i in range(0, n - min_bars + 1, start_step)]
    if not starts:
        raise ValueError("No start points available; reduce --min-bars or --start-step.")


    rs_total = int(len(cfgs))
    rs_done = int(len(already_done)) if accumulate_ok else 0
    rs_windows_per_cfg = int(len(starts))
    rs_windows_total = int(rs_total * rs_windows_per_cfg)
    rs_last_emit = 0
    t_prog0 = float(__import__("time").time())
    progress.write(
        {
            "stage": "rolling_starts",
            "phase": "run",
            "done": 0,
            "total": rs_total,
            "best_metric": "median_twr_total_return",
            "windows_per_cfg": rs_windows_per_cfg,
            "windows_total": rs_windows_total,
        }
    )


    # Progress bar (optional) / Parallelization
    jobs = int(max(1, int(getattr(args, "jobs", 1) or 1)))
    cfgs_run = [(cid, cfg, _norm) for (cid, cfg, _norm) in cfgs if not (accumulate_ok and str(cid) in already_done)]
    rs_done0 = int(rs_done)
    emit_every = int(max(1, int(getattr(args, "progress_every", 1) or 1)))

    details: List[Dict[str, Any]] = []
    best_score: Optional[float] = None
    recent_best: List[Tuple[str, float]] = []

    if jobs <= 1 or len(cfgs_run) <= 1:
        it_cfgs = cfgs_run
        if not args.no_progress:
            try:
                from tqdm import tqdm
                it_cfgs = tqdm(cfgs_run, total=len(cfgs_run), desc="RollingStarts", unit="cfg")
            except Exception:
                pass

        for cid, cfg, _norm in it_cfgs:
            rows_for_cfg: List[Dict[str, Any]] = []

            for s in starts:
                df_w = df_feat.iloc[int(s):]
                if len(df_w) < min_bars:
                    continue

                # IMPORTANT: new strategy instance per window (no state leakage)
                strategy = _instantiate_template(template_cls, cfg)

                metrics, _fills, eq_df, _trades, _guard = run_backtest_once(
                    df=df_w,
                    strategy=strategy,
                    seed=int(args.seed),
                    starting_equity=float(args.starting_equity),
                    constraints=constraints,
                    cfg=engine_cfg,
                    show_progress=False,
                    features_ready=True,
                    record_fills=False,
                    record_equity_curve=True,
                )

                eq = metrics.get("equity", {}) if isinstance(metrics, dict) else {}
                perf = _build_cashflow_performance_stats(df_w, eq_df)

                uw_days = _max_underwater_days(eq_df["equity"]) if (eq_df is not None and "equity" in eq_df.columns) else 0.0
                util_mean = _util_mean(eq_df)

                row = {
                    "config_id": str(cid),
                    "start_i": int(s),
                    "start_dt": str(df_feat["dt"].iloc[int(s)]) if "dt" in df_feat.columns else "",
                    "bars": int(len(df_w)),
                    "equity.end": float(eq.get("end", np.nan)),
                    "equity.cashflow_total": float(eq.get("cashflow_total", np.nan)),
                    "equity.net_profit_ex_cashflows": float(eq.get("net_profit_ex_cashflows", np.nan)),
                    "performance.twr_total_return": float(perf.get("twr_total_return", np.nan)),
                    "performance.annualized_return": float(perf.get("annualized_return", np.nan)),
                    "performance.max_drawdown_equity": float(perf.get("max_drawdown_equity", np.nan)),
                    "uw_max_days": float(uw_days),
                    "util_mean": float(util_mean),
                }
                rows_for_cfg.append(row)
                details.append(row)

            # Per-config score: median TWR across starts
            try:
                _vals = np.array([float(r.get("performance.twr_total_return", np.nan)) for r in rows_for_cfg], dtype=float)
                cfg_score = float(np.nanmedian(_vals)) if _vals.size else float("nan")
            except Exception:
                cfg_score = float("nan")

            if best_score is None or (np.isfinite(cfg_score) and cfg_score > float(best_score)):
                best_score = float(cfg_score)
            if np.isfinite(cfg_score):
                recent_best.append((str(cid), float(cfg_score)))
                recent_best = sorted(recent_best, key=lambda x: x[1], reverse=True)[:5]

            rs_done += 1

            if progress.enabled and ((rs_done - rs_done0) % emit_every == 0 or rs_done >= rs_total):
                elapsed = float(time.time() - t_prog0)
                progress.write(
                    {
                        "stage": "rolling_starts",
                        "phase": "run",
                        "done": int(rs_done),
                        "total": int(rs_total),
                        "rate": float((rs_done - rs_done0) / max(1e-9, elapsed)),
                        "windows_per_cfg": int(rs_windows_per_cfg),
                        "windows_done": int(rs_done * rs_windows_per_cfg),
                        "windows_total": int(rs_windows_total),
                        "best": (float(best_score) if (best_score is not None and np.isfinite(best_score)) else None),
                        "top": list(recent_best),
                        "config_id": str(cid),
                        "cfg_score": (float(cfg_score) if np.isfinite(cfg_score) else None),
                        "jobs": int(jobs),
                        "message": (f"cfg {str(cid)} score={cfg_score:.4g}" if np.isfinite(cfg_score) else f"cfg {str(cid)}"),
                    }
                )

    else:
        # Parallel across configs (each worker runs all start points for one config)
        df_feat_path = run_dir / "df_feat.parquet"
        if not df_feat_path.exists():
            try:
                df_feat.to_parquet(df_feat_path, index=False)
            except Exception:
                pass

        payloads: List[Tuple[str, Dict[str, Any]]] = []
        for cid, cfg, _norm in cfgs_run:
            payloads.append(
                (
                    str(cid),
                    {
                        "strategy_name": getattr(cfg, "strategy_name", "dca_swing"),
                        "side": getattr(cfg, "side", "long"),
                        "params": dict(getattr(cfg, "params", {}) or {}),
                    },
                )
            )

        max_workers = int(min(int(jobs), max(1, len(payloads))))
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_init,
            initargs=(
                str(df_feat_path),
                str(args.template),
                list(starts),
                int(min_bars),
                int(args.seed),
                float(args.starting_equity),
            ),
        ) as ex:
            futs = {ex.submit(_process_one_config, p): p[0] for p in payloads}

            for fut in as_completed(futs):
                cid = futs.get(fut, "")
                try:
                    cid2, rows_for_cfg, cfg_score = fut.result()
                except Exception as e:
                    rs_done += 1
                    progress.write(
                        {
                            "stage": "rolling_starts",
                            "phase": "run",
                            "done": int(rs_done),
                            "total": int(rs_total),
                            "jobs": int(max_workers),
                            "error": f"{cid}: {e}",
                        }
                    )
                    continue

                details.extend(list(rows_for_cfg))

                if best_score is None or (np.isfinite(cfg_score) and cfg_score > float(best_score)):
                    best_score = float(cfg_score)
                if np.isfinite(cfg_score):
                    recent_best.append((str(cid2), float(cfg_score)))
                    recent_best = sorted(recent_best, key=lambda x: x[1], reverse=True)[:5]

                rs_done += 1

                if progress.enabled and ((rs_done - rs_done0) % emit_every == 0 or rs_done >= rs_total):
                    elapsed = float(time.time() - t_prog0)
                    progress.write(
                        {
                            "stage": "rolling_starts",
                            "phase": "run",
                            "done": int(rs_done),
                            "total": int(rs_total),
                            "rate": float((rs_done - rs_done0) / max(1e-9, elapsed)),
                            "windows_per_cfg": int(rs_windows_per_cfg),
                            "windows_done": int(rs_done * rs_windows_per_cfg),
                            "windows_total": int(rs_windows_total),
                            "best": (float(best_score) if (best_score is not None and np.isfinite(best_score)) else None),
                            "top": list(recent_best),
                            "config_id": str(cid2),
                            "cfg_score": (float(cfg_score) if np.isfinite(cfg_score) else None),
                            "jobs": int(max_workers),
                            "message": (f"cfg {str(cid2)} score={cfg_score:.4g}" if np.isfinite(cfg_score) else f"cfg {str(cid2)}"),
                        }
                    )

    progress.write({"stage": "rolling_starts", "phase": "done", "done": int(rs_done), "total": int(rs_total), "jobs": int(jobs)})

    df_new_detail = pd.DataFrame(details)

    # Combine with previous detail if accumulating
    if accumulate_ok and (not df_prev_detail.empty):
        df_all_detail = pd.concat([df_prev_detail, df_new_detail], ignore_index=True)
    else:
        df_all_detail = df_new_detail

    if not df_all_detail.empty:
        # Deduplicate on (config_id, start_i)
        if "config_id" in df_all_detail.columns and "start_i" in df_all_detail.columns:
            df_all_detail["config_id"] = df_all_detail["config_id"].astype(str)
            df_all_detail["start_i"] = pd.to_numeric(df_all_detail["start_i"], errors="coerce").fillna(-1).astype(int)
            df_all_detail = df_all_detail.drop_duplicates(subset=["config_id", "start_i"], keep="last")
            try:
                df_all_detail = df_all_detail.sort_values(["config_id", "start_i"]).reset_index(drop=True)
            except Exception:
                pass

    df_sum = _summarize_from_detail(df_all_detail) if not df_all_detail.empty else pd.DataFrame([])

    df_all_detail.to_csv(detail_path, index=False)
    df_sum.to_csv(summary_path, index=False)
    meta_path.write_text(json.dumps(meta_now, indent=2), encoding="utf-8")

    print(f"Wrote: {detail_path}")
    print(f"Wrote: {summary_path}")
    progress.write({"stage": "rolling_starts", "phase": "done", "done": int(rs_done), "total": int(rs_total), "out_dir": str(out_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
