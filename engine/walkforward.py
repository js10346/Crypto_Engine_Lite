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

from engine.backtester import (
    BacktestConfig,
    _build_cashflow_performance_stats,
    _build_performance_stats,
    _efficiency_stats,
    _trade_stats,
    run_backtest_once,
)

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
_WF_OUT_DIR: Optional[Path] = None
_WF_WRITE_STITCH: bool = True
_WF_BAR_SEC: float = 86_400.0



# ============================================================
# Progress (optional): JSONL progress events for UI monitoring
# ============================================================

class ProgressWriter:
    """Append-only JSONL progress writer (safe no-op if path is None)."""

    def __init__(self, path: Optional[str]):
        self.enabled = bool(path)
        self.path: Optional[Path] = Path(path).resolve() if path else None
        self.t0 = time.time()
        self._fh = None
        if self.path:
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
        t = time.time()
        payload.setdefault("t", t)
        payload.setdefault("t_rel", float(t - self.t0))
        try:
            self._fh.write(json.dumps(payload) + "\n")
            self._fh.flush()
        except Exception:
            pass

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
    """
    Build walk-forward windows by timestamp.

    We also precompute a non-overlapping "segment" end inside each window:
      segment = [start, start + step_days)
    This is used to build a stitched curve without double-counting overlap.
    """
    if "ts" not in df_feat.columns or "dt" not in df_feat.columns:
        raise ValueError("df_feat must include 'ts' and 'dt' columns")

    ts = pd.to_numeric(df_feat["ts"], errors="coerce").dropna().astype(np.int64).to_numpy()
    if len(ts) < int(min_bars):
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
            i_step = int(np.searchsorted(ts_sorted, start + step_ms, side="left"))
            i_step = int(min(i_step, i1))  # clamp inside window
            seg_bars = int(max(0, i_step - i0))
            # if step is tiny or irregular ts, still guarantee at least 1 bar so stitching can progress
            if seg_bars <= 0:
                i_step = int(min(i0 + 1, i1))
                seg_bars = int(max(0, i_step - i0))

            out.append(
                {
                    "window_idx": int(k),
                    "start_i": int(i0),
                    "end_i": int(i1),
                    "step_end_i": int(i_step),  # exclusive
                    "start_ts": int(ts_sorted[i0]),
                    "end_ts_excl": int(start + window_ms),
                    "step_end_ts_excl": int(start + step_ms),
                    "start_dt": str(df_feat["dt"].iloc[i0]),
                    "end_dt": str(df_feat["dt"].iloc[i1 - 1]),
                    "step_end_dt": str(df_feat["dt"].iloc[i_step - 1]) if i_step - 1 >= i0 else str(df_feat["dt"].iloc[i0]),
                    "bars": int(bars),
                    "segment_bars": int(seg_bars),
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
    market_mode: str,
    out_dir: str,
    write_stitch: bool = True,
) -> None:
    global _WF_DF_FEAT, _WF_WINDOWS, _WF_TEMPLATE_CLS, _WF_CONSTRAINTS, _WF_ENGINE_CFG, _WF_OUT_DIR, _WF_WRITE_STITCH

    _WF_DF_FEAT = pd.read_parquet(df_feat_path)
    # Infer bar seconds from ts spacing (fallback daily)
    try:
        ts = pd.to_numeric(_WF_DF_FEAT.get('ts'), errors='coerce').dropna().astype(np.int64)
        if len(ts) >= 3:
            d = ts.diff().dropna().to_numpy(dtype=np.int64)
            med_ms = float(np.median(d)) if len(d) else 86_400_000.0
            globals()['_WF_BAR_SEC'] = float(max(1.0, med_ms / 1000.0))
        else:
            globals()['_WF_BAR_SEC'] = 86_400.0
    except Exception:
        globals()['_WF_BAR_SEC'] = 86_400.0
    _WF_WINDOWS = list(windows)
    _WF_TEMPLATE_CLS = _import_symbol(template_dotted)
    _WF_CONSTRAINTS = _make_constraints()
    _WF_ENGINE_CFG = BacktestConfig(market_mode=str(market_mode or "spot").lower())
    _WF_OUT_DIR = Path(out_dir).resolve()
    _WF_WRITE_STITCH = bool(write_stitch)


def _max_underwater_bars(x: np.ndarray) -> int:
    """Longest consecutive streak below prior peak."""
    if x is None or len(x) == 0:
        return 0
    peak = -float("inf")
    cur = 0
    best = 0
    for v in x:
        if v > peak:
            peak = float(v)
        if v < peak - 1e-12:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def _compute_twr_index(eq_df: pd.DataFrame) -> np.ndarray:
    """
    Return the TWR index series (deposit-neutral) from an equity curve that includes 'equity' and optional 'cashflow'.
    idx[0] == 1.0.
    """
    eq = pd.to_numeric(eq_df.get("equity", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
    cf = pd.to_numeric(eq_df.get("cashflow", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
    n = int(len(eq))
    if n <= 0:
        return np.ones(0, dtype=np.float64)
    idx = np.ones(n, dtype=np.float64)
    for t in range(1, n):
        denom = float(eq[t - 1] + cf[t])
        if denom <= 1e-12 or (not math.isfinite(denom)):
            idx[t] = idx[t - 1]
            continue
        r = float(eq[t] / denom)
        if (not math.isfinite(r)) or r <= 0.0:
            r = 1.0
        idx[t] = idx[t - 1] * r
    return idx


def _run_one_window(
    df_window: pd.DataFrame,
    cfg: StrategyConfig,
    *,
    seed: int,
    starting_equity: float,
    constraints: EngineConstraints,
    engine_cfg: BacktestConfig,
    segment_bars: int,
) -> Dict[str, Any]:
    """
    Run a single walk-forward window.

    Spot-only for SaaS. We always record an equity curve (needed for cashflow-aware TWR).
    We also return a *non-overlapping segment* from the window (first step-days),
    which is used to build a stitched curve without overlap.
    """
    strategy = _instantiate_template(_WF_TEMPLATE_CLS, cfg)

    is_spot = str(getattr(engine_cfg, "market_mode", "spot")).lower() == "spot"

    metrics, _fills_df, eq_df, trades_df, _guard = run_backtest_once(
        df=df_window,
        strategy=strategy,
        seed=int(seed),
        starting_equity=float(starting_equity),
        constraints=constraints,
        cfg=engine_cfg,
        show_progress=False,
        features_ready=True,
        record_fills=False,
        record_equity_curve=True,
    )

    # Attach ts (eq_df doesn't include it)
    if eq_df is None or eq_df.empty:
        eq_df = pd.DataFrame({"dt": df_window["dt"].astype(str).to_numpy(), "equity": np.nan})
    try:
        eq_df = eq_df.copy()
        if "ts" not in eq_df.columns and "ts" in df_window.columns and len(eq_df) == len(df_window):
            eq_df["ts"] = pd.to_numeric(df_window["ts"], errors="coerce").to_numpy()
    except Exception:
        pass

    # Performance stats
    if is_spot:
        perf = _build_cashflow_performance_stats(df_window, eq_df)
        total_return = float(perf.get("twr_total_return", 0.0) or 0.0)
    else:
        perf = _build_performance_stats(df_window, eq_df)
        total_return = float(perf.get("total_return", 0.0) or 0.0)

    # Trade/efficiency
    tstats = _trade_stats(trades_df)
    efficiency = _efficiency_stats(trades_df)

    # Underwater (on TWR index for spot; on equity index otherwise)
    bar_sec = float(perf.get("bar_seconds", 86_400.0) or 86_400.0)
    if is_spot:
        twr_idx = _compute_twr_index(eq_df)
        uw_bars = _max_underwater_bars(twr_idx)
        uw_days = float(uw_bars) * (bar_sec / 86_400.0)
    else:
        eq_series = pd.to_numeric(eq_df.get("equity", 0.0), errors="coerce").fillna(0.0).astype(float).to_numpy()
        uw_bars = _max_underwater_bars(eq_series)
        uw_days = float(uw_bars) * (bar_sec / 86_400.0)
        twr_idx = np.ones(len(eq_series), dtype=np.float64)

    # Window drawdown (deposit-neutral for spot)
    window_mdd = float(perf.get("max_drawdown", 0.0) or 0.0)

    # Non-overlapping segment (first step period inside the window)
    seg_n = int(max(0, min(int(segment_bars), len(eq_df))))
    if seg_n <= 1:
        seg_df = eq_df.iloc[:0].copy()
        seg_return = 0.0
        seg_mdd = 0.0
        seg_uw_days = 0.0
    else:
        seg_df = eq_df.iloc[:seg_n].copy()
        seg_idx = twr_idx[:seg_n] if len(twr_idx) >= seg_n else np.ones(seg_n, dtype=np.float64)
        # normalize segment to 1.0 at start
        if len(seg_idx):
            seg_idx = seg_idx / float(seg_idx[0] if seg_idx[0] != 0 else 1.0)
        seg_df["twr_idx"] = seg_idx.astype(float)
        seg_return = float(seg_idx[-1] - 1.0) if len(seg_idx) else 0.0
        # drawdown/underwater inside segment
        # (max drawdown on index; underwater on index)
        seg_mdd = float(_build_cashflow_performance_stats(df_window.iloc[:seg_n], seg_df).get("max_drawdown", 0.0) or 0.0) if is_spot else 0.0
        seg_uw_bars = _max_underwater_bars(seg_idx)
        seg_uw_days = float(seg_uw_bars) * (bar_sec / 86_400.0)

    eq = metrics.get("equity", {}) if isinstance(metrics, dict) else {}
    # Force explicit fields we care about
    eq = dict(eq)
    eq["total_return"] = float(total_return)
    eq["max_drawdown"] = float(window_mdd)
    eq["underwater_days"] = float(uw_days)

    return {
        "equity": eq,
        "perf": perf,
        "trades_summary": tstats,
        "efficiency": efficiency,
        "underwater_bars": int(uw_bars),
        "underwater_days": float(uw_days),
        "segment_df": seg_df,  # used for stitching
        "segment_return": float(seg_return),
        "segment_mdd": float(seg_mdd),
        "segment_underwater_days": float(seg_uw_days),
    }

def _wf_run_config_chunk(
    chunk: List[Tuple[str, StrategyConfig, str, str]],
    seed: int,
    starting_equity: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (wf_rows, wf_summaries) for this chunk.

    Also writes stitched curves per-config when enabled.
    """
    if (
        _WF_DF_FEAT is None
        or _WF_WINDOWS is None
        or _WF_CONSTRAINTS is None
        or _WF_ENGINE_CFG is None
        or _WF_OUT_DIR is None
    ):
        raise RuntimeError("WF worker not initialized")

    df_feat = _WF_DF_FEAT
    windows = _WF_WINDOWS
    constraints = _WF_CONSTRAINTS
    engine_cfg = _WF_ENGINE_CFG
    out_dir = _WF_OUT_DIR

    stitched_dir = out_dir / "stitched"
    if _WF_WRITE_STITCH:
        stitched_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for config_id, cfg, strategy_name, side in chunk:
        window_returns: List[float] = []
        window_mdds: List[float] = []
        window_uw_days: List[float] = []
        window_trades: List[int] = []

        seg_returns: List[float] = []
        seg_mdds: List[float] = []
        seg_uw_days: List[float] = []

        stitched_parts: List[pd.DataFrame] = []
        stitched_scale = 1.0

        for w in windows:
            i0 = int(w["start_i"])
            i1 = int(w["end_i"])
            seg_bars = int(w.get("segment_bars", max(0, int(w.get("step_end_i", i0)) - i0)) or 0)

            df_w = df_feat.iloc[i0:i1]

            res = _run_one_window(
                df_window=df_w,
                cfg=cfg,
                seed=int(seed),
                starting_equity=float(starting_equity),
                constraints=constraints,
                engine_cfg=engine_cfg,
                segment_bars=seg_bars,
            )

            eq = res["equity"]
            tstats = res["trades_summary"]

            r = float(eq.get("total_return", 0.0) or 0.0)
            mdd = float(eq.get("max_drawdown", 0.0) or 0.0)
            uwd = float(eq.get("underwater_days", 0.0) or 0.0)
            ntr = int(tstats.get("trades_closed", 0) or 0)

            window_returns.append(r)
            window_mdds.append(mdd)
            window_uw_days.append(uwd)
            window_trades.append(ntr)

            seg_r = float(res.get("segment_return", 0.0) or 0.0)
            seg_d = float(res.get("segment_mdd", 0.0) or 0.0)
            seg_u = float(res.get("segment_underwater_days", 0.0) or 0.0)

            seg_returns.append(seg_r)
            seg_mdds.append(seg_d)
            seg_uw_days.append(seg_u)

            # Failure heuristics (NOT filters; just story aides)
            reasons: List[str] = []
            if ntr <= 0:
                reasons.append("no_trades")
            if r < 0.0:
                reasons.append("neg_return")
            if mdd >= 0.35:
                reasons.append("dd>=35%")
            if uwd >= (0.75 * float(w.get("window_days", 0) or 0) if "window_days" in w else 999999):
                # if window_days isn't present, this won't trigger
                reasons.append("long_underwater")

            all_rows.append(
                {
                    "config_id": str(config_id),
                    "strategy_name": str(strategy_name),
                    "side": str(side),
                    "window_idx": int(w["window_idx"]),
                    "window_start_dt": str(w["start_dt"]),
                    "window_end_dt": str(w["end_dt"]),
                    "step_end_dt": str(w.get("step_end_dt", "")),
                    "bars": int(w["bars"]),
                    "segment_bars": int(w.get("segment_bars", 0) or 0),
                    "window_return": float(r),
                    "window_max_drawdown": float(mdd),
                    "window_underwater_days": float(uwd),
                    "segment_return": float(seg_r),
                    "segment_max_drawdown": float(seg_d),
                    "segment_underwater_days": float(seg_u),
                    "trades_closed": int(ntr),
                    "win_rate": float(tstats.get("win_rate", 0.0) or 0.0),
                    "profit_factor": float(tstats.get("profit_factor", 0.0) or 0.0),
                    "expectancy": float(tstats.get("expectancy", 0.0) or 0.0),
                    "fee_impact_pct": float(res["efficiency"].get("fee_impact_pct", 0.0) or 0.0),
                    "flags": ";".join(reasons),
                }
            )

            # Build stitched curve from the (non-overlapping) segment
            if _WF_WRITE_STITCH:
                seg_df: pd.DataFrame = res.get("segment_df")
                if seg_df is not None and not seg_df.empty and "twr_idx" in seg_df.columns:
                    part = seg_df[["dt", "ts", "twr_idx"]].copy() if "ts" in seg_df.columns else seg_df[["dt", "twr_idx"]].copy()
                    part["stitched_twr"] = pd.to_numeric(part["twr_idx"], errors="coerce").fillna(1.0).astype(float) * float(stitched_scale)
                    part["config_id"] = str(config_id)
                    part["window_idx"] = int(w["window_idx"])
                    stitched_parts.append(part[["config_id", "window_idx", "dt"] + (["ts"] if "ts" in part.columns else []) + ["stitched_twr"]])
                    try:
                        stitched_scale = float(part["stitched_twr"].iloc[-1])
                    except Exception:
                        pass

        # Summary across windows
        arr_r = np.array(window_returns, dtype=float)
        arr_dd = np.array(window_mdds, dtype=float)
        arr_uw = np.array(window_uw_days, dtype=float)
        arr_tr = np.array(window_trades, dtype=float)

        n = int(len(arr_r))
        if n == 0:
            continue

        def q(a: np.ndarray, p: float) -> float:
            if a is None or len(a) == 0:
                return 0.0
            a2 = a[np.isfinite(a)]
            if len(a2) == 0:
                return 0.0
            return float(np.quantile(a2, p))

        # Trades stability
        tr_mean = float(np.mean(arr_tr)) if len(arr_tr) else 0.0
        tr_std = float(np.std(arr_tr)) if len(arr_tr) else 0.0
        tr_cv = float(tr_std / tr_mean) if tr_mean > 1e-12 else 0.0

        # Stitched performance (segment-based compound) if available
        stitched_total_return = None
        stitched_mdd = None
        stitched_uw_days = None
        stitched_path = None
        if _WF_WRITE_STITCH and stitched_parts:
            stitched_df = pd.concat(stitched_parts, ignore_index=True)
            # Ensure monotonic time for plotting
            if "ts" in stitched_df.columns:
                stitched_df = stitched_df.sort_values(["ts", "window_idx"], kind="mergesort")
            else:
                stitched_df = stitched_df.sort_values(["dt", "window_idx"], kind="mergesort")

            stitched_path = stitched_dir / f"{str(config_id)}.csv"
            stitched_df.to_csv(stitched_path, index=False)

            st_series = pd.to_numeric(stitched_df["stitched_twr"], errors="coerce").ffill().fillna(1.0).astype(float).to_numpy()
            if len(st_series) >= 2:
                stitched_total_return = float(st_series[-1] - 1.0)
                # drawdown + underwater on stitched index
                # max drawdown
                peak = -float("inf")
                mdd_best = 0.0
                for v in st_series:
                    if v > peak:
                        peak = float(v)
                    dd = (peak - float(v)) / peak if peak > 1e-12 else 0.0
                    if dd > mdd_best:
                        mdd_best = dd
                stitched_mdd = float(mdd_best)
                stitched_uw_bars = _max_underwater_bars(st_series)
                # bar_seconds is not directly available here; use 1 bar == 1 unit (UI can interpret)
                stitched_uw_days = float(stitched_uw_bars) * (float(globals().get('_WF_BAR_SEC', 86_400.0)) / 86_400.0)

        summaries.append(
            {
                "config_id": str(config_id),
                "strategy_name": str(strategy_name),
                "side": str(side),
                "windows": int(n),
                "pct_profitable_windows": float((arr_r > 0.0).mean()) if n else 0.0,

                "return_p10": q(arr_r, 0.10),
                "return_p50": q(arr_r, 0.50),
                "return_p90": q(arr_r, 0.90),

                "dd_p10": q(arr_dd, 0.10),
                "dd_p50": q(arr_dd, 0.50),
                "dd_p90": q(arr_dd, 0.90),

                "uw_days_p10": q(arr_uw, 0.10),
                "uw_days_p50": q(arr_uw, 0.50),
                "uw_days_p90": q(arr_uw, 0.90),

                "trades_p10": q(arr_tr, 0.10),
                "trades_p50": q(arr_tr, 0.50),
                "trades_p90": q(arr_tr, 0.90),
                "pct_windows_traded": float((arr_tr >= 1.0).mean()) if n else 0.0,
                "trades_cv": float(tr_cv),

                # Back-compat with the previous UI columns:
                "mean_window_return": float(np.mean(arr_r)) if n else 0.0,
                "median_window_return": float(np.median(arr_r)) if n else 0.0,
                "min_window_return": float(np.min(arr_r)) if n else 0.0,
                "max_window_return": float(np.max(arr_r)) if n else 0.0,
                "mean_trades_per_window": float(np.mean(arr_tr)) if n else 0.0,
                "min_trades_per_window": float(np.min(arr_tr)) if n else 0.0,
                "max_trades_per_window": float(np.max(arr_tr)) if n else 0.0,

                # Segment (stitched) summary
                "stitched_total_return": (float(stitched_total_return) if stitched_total_return is not None else None),
                "stitched_max_drawdown": (float(stitched_mdd) if stitched_mdd is not None else None),
                "stitched_underwater_days": (float(stitched_uw_days) if stitched_uw_days is not None else None),
                "stitched_path": (f"stitched/{Path(stitched_path).name}" if stitched_path is not None else None),
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
    ap.add_argument("--progress-file", default=None, help="Optional JSONL progress file for UI monitoring.")
    ap.add_argument("--progress-every", type=int, default=10, help="Emit progress every N configs (approx).")
    ap.add_argument(
        "--write-stitch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-config stitched walk-forward curves (non-overlapping segments) to out_dir/stitched/.",
    )
    ap.add_argument("--out", default=None, help="Output folder (default: under from-run)")
    ap.add_argument(
        "--market-mode",
        default=None,
        choices=["spot", "perps"],
        help="Override market mode (default: spot or inferred from batch_meta.json).",
    )
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
    market_mode = (
        str(args.market_mode) if args.market_mode else str(meta.get("market_mode", "spot"))
    )

    sort_by_default, sort_desc_default = _infer_sort_defaults(meta)
    sort_by = str(args.sort_by) if args.sort_by else sort_by_default
    sort_desc = bool(args.sort_desc) if args.sort_desc is not None else bool(sort_desc_default)

    # Output folder
    tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out).resolve() if args.out else (run_dir / f"walkforward_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    progress = ProgressWriter(args.progress_file)
    progress.write({"stage": "walkforward", "phase": "init", "done": 0, "total": 0, "out_dir": str(out_dir)})


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
        "write_stitch": bool(getattr(args, "write_stitch", True)),
        "stitched_dir": str(Path(out_dir) / "stitched"),
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

    wf_total = int(len(tasks))
    wf_done = 0
    wf_last_emit = 0
    wf_best_median: Optional[float] = None
    wf_best_pct: Optional[float] = None
    t_prog0 = time.time()
    progress.write(
        {
            "stage": "walkforward",
            "phase": "run",
            "done": 0,
            "total": wf_total,
            "windows": int(len(windows)),
            "rows_total": int(wf_total * len(windows)),
        }
    )


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
        _wf_worker_init(str(df_feat_path), str(template), windows, str(market_mode), str(out_dir), bool(args.write_stitch))
        for ch in chunks:
            rows, sums = _wf_run_config_chunk(
                ch,
                seed=int(seed),
                starting_equity=float(starting_equity),
            )
            wf_rows.extend(rows)
            wf_summaries.extend(sums)
            wf_done += int(len(ch))
            for srow in sums:
                try:
                    mv = float(srow.get("median_window_return", float("nan")))
                    if math.isfinite(mv):
                        wf_best_median = mv if wf_best_median is None else max(float(wf_best_median), mv)
                except Exception:
                    pass
                try:
                    pv = float(srow.get("pct_profitable_windows", float("nan")))
                    if math.isfinite(pv):
                        wf_best_pct = pv if wf_best_pct is None else max(float(wf_best_pct), pv)
                except Exception:
                    pass

            pe = int(getattr(args, "progress_every", 10) or 10)
            if progress.enabled and ((wf_done - wf_last_emit) >= pe or wf_done >= wf_total):
                wf_last_emit = int(wf_done)
                progress.write(
                    {
                        "stage": "walkforward",
                        "phase": "run",
                        "done": int(wf_done),
                        "total": int(wf_total),
                        "rate": float(wf_done / max(1e-9, (time.time() - t_prog0))),
                        "windows": int(len(windows)),
                        "rows_done": int(wf_done * len(windows)),
                        "rows_total": int(wf_total * len(windows)),
                        "best": (float(wf_best_median) if wf_best_median is not None else None),
                        "best_metric": "median_window_return",
                        "best_pct": (float(wf_best_pct) if wf_best_pct is not None else None),
                        "best_detail": {
                            "median_window_return": (float(wf_best_median) if wf_best_median is not None else None),
                            "pct_profitable_windows": (float(wf_best_pct) if wf_best_pct is not None else None),
                        },
                    }
                )

            if use_tqdm and pbar is not None:
                pbar.update(len(ch))
    else:
        with ProcessPoolExecutor(
            max_workers=jobs,
            initializer=_wf_worker_init,
            initargs=(str(df_feat_path), str(template), windows, str(market_mode), str(out_dir), bool(args.write_stitch)),
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

                n_done = int(fut_to_n.get(fut, 1))
                wf_done += n_done
                for srow in sums:
                    try:
                        mv = float(srow.get("median_window_return", float("nan")))
                        if math.isfinite(mv):
                            wf_best_median = mv if wf_best_median is None else max(float(wf_best_median), mv)
                    except Exception:
                        pass
                    try:
                        pv = float(srow.get("pct_profitable_windows", float("nan")))
                        if math.isfinite(pv):
                            wf_best_pct = pv if wf_best_pct is None else max(float(wf_best_pct), pv)
                    except Exception:
                        pass

                pe = int(getattr(args, "progress_every", 10) or 10)
                if progress.enabled and ((wf_done - wf_last_emit) >= pe or wf_done >= wf_total):
                    wf_last_emit = int(wf_done)
                    progress.write(
                        {
                            "stage": "walkforward",
                            "phase": "run",
                            "done": int(wf_done),
                            "total": int(wf_total),
                            "rate": float(wf_done / max(1e-9, (time.time() - t_prog0))),
                            "windows": int(len(windows)),
                            "rows_done": int(wf_done * len(windows)),
                            "rows_total": int(wf_total * len(windows)),
                            "best": {
                                "median_window_return": float(wf_best_median) if wf_best_median is not None else None,
                                "pct_profitable_windows": float(wf_best_pct) if wf_best_pct is not None else None,
                            },
                        }
                    )

                if use_tqdm and pbar is not None:
                    pbar.update(n_done)

    if use_tqdm and pbar is not None:
        pbar.close()

    elapsed = time.time() - t_run0

    # Save
    wf_rows_df = pd.DataFrame(wf_rows)
    wf_sum_df = pd.DataFrame(wf_summaries)

    wf_rows_df.to_csv(out_dir / "wf_results.csv", index=False)
    wf_sum_df.to_csv(out_dir / "wf_summary.csv", index=False)
    # Helpful story file: windows that triggered any heuristic flags
    try:
        if "flags" in wf_rows_df.columns:
            wf_fail = wf_rows_df[wf_rows_df["flags"].astype(str).str.len() > 0].copy()
            wf_fail.to_csv(out_dir / "wf_failures.csv", index=False)
    except Exception:
        pass

    progress.write({"stage": "walkforward", "phase": "done", "done": int(wf_done), "total": int(wf_total), "out_dir": str(out_dir)})
    print(f"\nWalk-forward complete. Output: {out_dir}")
    print(f"Configs: {len(tasks)}   Windows: {len(windows)}")
    print(f"Rows:    {len(wf_rows_df)} (config x window)")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Files:   wf_results.csv, wf_summary.csv, wf_failures.csv, wf_meta.json")
    if bool(getattr(args, "write_stitch", True)):
        print(f"         stitched/ (per-config stitched curves)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())