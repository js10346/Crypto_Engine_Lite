
from __future__ import annotations

import json
import zipfile
import io
import hashlib
import os
import shutil
import math
import re
import subprocess
import sys
import time
import threading
import queue
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
com = components

# Optional: used for Build-step “reality check” stats (TA filters)
try:
    from engine.features import add_features as _add_features
except Exception:  # pragma: no cover
    _add_features = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover
    px = None
    go = None
    make_subplots = None

# Plotly rendering config (Streamlit wants config dict for Plotly options)
PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
}

# Plotly chart helper: Streamlit has been changing its API (width vs use_container_width).
# We pick the supported signature at runtime to avoid deprecation spam.
import inspect as _inspect

try:
    _PLOTLY_CHART_SIG = _inspect.signature(st.plotly_chart)
except Exception:  # pragma: no cover
    _PLOTLY_CHART_SIG = None

_PLOTLY_HAS_WIDTH = bool(_PLOTLY_CHART_SIG and ("width" in _PLOTLY_CHART_SIG.parameters))
_PLOTLY_HAS_UCW = bool(_PLOTLY_CHART_SIG and ("use_container_width" in _PLOTLY_CHART_SIG.parameters))

def _plotly(fig, *, key: Optional[str] = None) -> None:
    kwargs: Dict[str, Any] = {"config": PLOTLY_CONFIG}
    if key is not None:
        kwargs["key"] = key
    if _PLOTLY_HAS_WIDTH:
        kwargs["width"] = "stretch"
    elif _PLOTLY_HAS_UCW:
        kwargs["use_container_width"] = True
    st.plotly_chart(fig, **kwargs)


# Optional (nice formatting + metric labels)
try:
    from lab.metrics import METRICS
except Exception:  # pragma: no cover
    METRICS = {}


# =============================================================================
# Visual system (Sprint 6)
# =============================================================================

PASS_COLOR = "#00C853"   # vibrant green
WARN_COLOR = "#FFD600"   # bright amber
FAIL_COLOR = "#FF1744"   # vivid red
NEUTRAL_COLOR = "#9E9E9E"
ACCENT_BLUE = "#2979FF"  # electric blue

VERDICT_COLORS = {
    "PASS": PASS_COLOR,
    "WARN": WARN_COLOR,
    "FAIL": FAIL_COLOR,
    "UNMEASURED": NEUTRAL_COLOR,
}

def _style_fig(fig, title: str | None = None):
    """Apply a consistent sleek + vibrant look to plotly figures."""
    if fig is None:
        return fig
    try:
        fig.update_layout(
            template="plotly_white",
            title=title or (fig.layout.title.text if getattr(fig.layout, "title", None) else None),
            margin=dict(l=20, r=20, t=85, b=20),
            font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial", size=13, color="#111827"),
            paper_bgcolor="white",
            plot_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="left", x=0, font=dict(size=12)),
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(17,24,39,0.08)", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(17,24,39,0.08)", zeroline=False)
    except Exception:
        pass
    return fig

def _verdict_color(v: str) -> str:
    return VERDICT_COLORS.get(str(v or "").upper(), NEUTRAL_COLOR)

# =============================================================================
# App meta
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "runs"
DATA_DIR = REPO_ROOT / "data"

PY = sys.executable

st.set_page_config(page_title="Spot Strategy Stress Lab", layout="wide")
st.title("Spot Strategy Stress Lab")
st.caption("Spot-only. Batch → Rolling Starts → Walkforward → Grand verdict.")

# =============================================================================
# Small utilities
# =============================================================================

def _now_slug() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _slug(s: Any, *, max_len: int = 140) -> str:
    s = str(s or "").strip()
    if not s:
        return "run"
    s = s.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s or "run")[: int(max_len)]


def _fmt_pct(x: Any, *, digits: int = 2) -> str:
    try:
        v = float(x)
        if not math.isfinite(v):
            return "n/a"
        return f"{v * 100:.{digits}f}%"
    except Exception:
        return "n/a"


def _fmt_num(x: Any, *, digits: int = 4) -> str:
    try:
        v = float(x)
        if not math.isfinite(v):
            return "n/a"
        return f"{v:.{digits}f}"
    except Exception:
        return "n/a"



def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _drawdown_to_frac(dd: pd.Series) -> pd.Series:
    """Best-effort: convert drawdown series to a 0..1 fraction."""
    x = _to_float_series(dd).copy()
    # Heuristic: if values look like 20..80 (percent) convert to fraction
    finite = x.dropna()
    if len(finite) > 0:
        q95 = float(finite.quantile(0.95))
        if q95 > 2.0:  # likely percent points
            x = x / 100.0
    return x


def _pareto_frontier(df: pd.DataFrame, x: str, y: str, *, x_round: int = 6, y_eps: float = 1e-12) -> pd.DataFrame:
    """Return Pareto frontier for maximize y, minimize x.

    Notes:
    - Collapses near-duplicate x values (rounding) to avoid ugly vertical segments.
    - Uses a strict-improvement threshold on y to avoid float jitter.
    """
    if df.empty:
        return df.copy()

    tmp = df[[x, y]].dropna().copy()
    if tmp.empty:
        return tmp

    tmp[x] = pd.to_numeric(tmp[x], errors="coerce")
    tmp[y] = pd.to_numeric(tmp[y], errors="coerce")
    tmp = tmp.dropna()
    if tmp.empty:
        return tmp

    # Collapse duplicate/near-duplicate x values so the frontier doesn't look like a glitchy barcode.
    tmp["_xbin"] = tmp[x].round(x_round)
    tmp = tmp.groupby("_xbin", as_index=False).agg({x: "min", y: "max"}).sort_values(x, ascending=True)

    best = -1e100
    keep_rows = []
    for _, row in tmp.iterrows():
        val = float(row[y])
        if val > best + y_eps:
            best = val
            keep_rows.append(row)

    out = pd.DataFrame(keep_rows)
    # Ensure x,y columns exist for downstream plotting.
    return out[[x, y]].sort_values(x, ascending=True)


def _pareto_frontier_rows(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    x_round: int = 6,
    y_eps: float = 1e-12,
) -> pd.DataFrame:
    """Return Pareto frontier *rows* from the original dataframe.

    We maximize y and minimize x. This version keeps the original row payload (config_id, label, trades, etc.)
    so we can show a frontier table and better hover text.

    Implementation:
    1) Round x into bins to collapse near-duplicates (avoids ugly vertical barcode segments).
    2) Within each x-bin, pick the row with max y.
    3) Sweep increasing x and keep only rows that strictly improve y (epsilon to avoid float jitter).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    tmp = df.copy()
    tmp[x] = pd.to_numeric(tmp[x], errors="coerce")
    tmp[y] = pd.to_numeric(tmp[y], errors="coerce")
    tmp = tmp.dropna(subset=[x, y])
    if tmp.empty:
        return pd.DataFrame()

    tmp["_xbin"] = tmp[x].round(x_round)

    # Pick best-y row per bin (if ties, idxmax returns first occurrence).
    try:
        idx = tmp.groupby("_xbin")[y].idxmax()
        cand = tmp.loc[idx].sort_values(x, ascending=True)
    except Exception:
        # Fallback: no groupby for weird data
        cand = tmp.sort_values([x, y], ascending=[True, False]).drop_duplicates("_xbin", keep="first")

    best = -1e100
    keep_rows = []
    for _, row in cand.iterrows():
        val = float(row[y])
        if val > best + y_eps:
            best = val
            keep_rows.append(row)

    out = pd.DataFrame(keep_rows).drop(columns=["_xbin"], errors="ignore")
    return out.sort_values(x, ascending=True)



def _goodness_percentile(s: pd.Series, *, low_is_good: bool) -> pd.Series:
    """0..1 where 1 is best."""
    x = _to_float_series(s)
    pct = x.rank(pct=True, ascending=True)
    if low_is_good:
        n = max(int(pct.notna().sum()), 1)
        return (1.0 - pct) + (1.0 / n)
    return pct


def _fmt_money(x: Any) -> str:
    try:
        v = float(x)
        if not math.isfinite(v):
            return "n/a"
        return f"{v:,.2f}"
    except Exception:
        return "n/a"


def _metric_label(metric_id: str) -> str:
    spec = METRICS.get(metric_id)
    return spec.label if spec else metric_id


def _metric_fmt(metric_id: str, x: Any) -> str:
    spec = METRICS.get(metric_id)
    if spec and hasattr(spec, "fmt"):
        try:
            return spec.fmt(float(x))
        except Exception:
            return str(x)
    # Fallback
    if "dd" in metric_id or "drawdown" in metric_id or metric_id.endswith("_return"):
        return _fmt_pct(x)
    return _fmt_num(x)


def _read_json(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

@st.cache_data(show_spinner=False)
def _read_csv_cached(path: str, mtime: float) -> pd.DataFrame:
    _ = mtime  # cache buster
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _add_features_cached(path: str, mtime: float) -> Optional[pd.DataFrame]:
    """Load CSV + compute features once per dataset version (used only for UI context)."""
    _ = mtime  # cache buster
    if _add_features is None:
        return None
    df = pd.read_csv(path)
    try:
        df.columns = [str(c).strip().lower() for c in df.columns]
    except Exception:
        pass
    need = {"open", "high", "low", "close"}
    if not need.issubset(set(df.columns)):
        return None
    return _add_features(df)


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return _read_csv_cached(str(path), path.stat().st_mtime)
    except Exception:
        return None


def _infer_bar_ms_from_csv(path: Path) -> Optional[int]:
    """Infer median bar size in milliseconds from the first ~5000 rows of a CSV.

    Tries 'ts' (ms epoch) first; falls back to parsing 'dt'/'date' as datetimes.
    Returns None if it can't infer a stable interval.
    """
    try:
        if not path.exists():
            return None
        sample = pd.read_csv(path, nrows=5000)
        if sample is None or len(sample) < 3:
            return None

        cols = {c.lower(): c for c in sample.columns}

        if "ts" in cols:
            ts = pd.to_numeric(sample[cols["ts"]], errors="coerce").dropna().astype("int64")
            if len(ts) < 3:
                return None
            diffs = ts.diff().dropna()
            med = float(diffs.median())
            if med <= 0:
                return None
            return int(med)

        for key in ["dt", "date", "datetime", "time", "timestamp"]:
            if key in cols:
                dt = pd.to_datetime(sample[cols[key]], errors="coerce", utc=True)
                dt = dt.dropna()
                if len(dt) < 3:
                    continue
                diffs = dt.diff().dropna().dt.total_seconds() * 1000.0
                med = float(diffs.median())
                if med <= 0:
                    continue
                return int(med)

        return None
    except Exception:
        return None


def _bars_per_day_from_run_meta(run_dir: Path) -> int:
    """Estimate bars/day based on the run's batch_meta.json and dataset."""
    try:
        meta = _read_json(run_dir / "batch_meta.json")
        data = meta.get("data") if isinstance(meta, dict) else None
        if not data:
            return 1
        bar_ms = _infer_bar_ms_from_csv(Path(str(data)))
        if not bar_ms or bar_ms <= 0:
            return 1
        bpd = int(round(86_400_000 / float(bar_ms)))
        return int(max(1, min(86_400, bpd)))
    except Exception:
        return 1


def _human_bar_interval_from_run(run_dir: Path) -> str:
    try:
        meta = _read_json(run_dir / "batch_meta.json")
        data = meta.get("data") if isinstance(meta, dict) else None
        if not data:
            return "unknown"
        bar_ms = _infer_bar_ms_from_csv(Path(str(data)))
        if not bar_ms:
            return "unknown"
        sec = bar_ms / 1000.0
        if sec >= 86_400:
            return f"~{sec/86_400:.1f} days/bar"
        if sec >= 3600:
            return f"~{sec/3600:.1f} hours/bar"
        if sec >= 60:
            return f"~{sec/60:.1f} minutes/bar"
        return f"~{sec:.0f} seconds/bar"
    except Exception:
        return "unknown"

@st.cache_data(show_spinner=False)
def _read_jsonl_cached(path: str, mtime: float) -> List[Dict[str, Any]]:
    _ = mtime
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                out.append(json.loads(s))
            except Exception:
                continue
    return out


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return _read_jsonl_cached(str(path), path.stat().st_mtime)


def _summarize_grid_composition(grid_path: Path) -> Dict[str, Any]:
    """Summarize what kind of population we generated (for the run-story 'dopamine loop').

    This is NOT advice; it's just a factual breakdown of the variant set.
    """
    from datetime import datetime as _dt

    total = 0
    logic = 0
    always = 0
    filter_counts = defaultdict(int)

    trail = 0
    time_stop = 0
    sl = 0
    tp = 0

    # Extra: characterize logic complexity (bounded + readable)
    logic_clauses_sum = 0
    logic_conds_sum = 0

    if not grid_path.exists():
        return {"total": 0, "generated_at": _dt.utcnow().isoformat()}

    with open(grid_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            total += 1
            params = row.get("params") or {}
            bf = str(params.get("buy_filter", "none") or "none").strip().lower()
            el = params.get("entry_logic") if isinstance(params.get("entry_logic"), dict) else None
            clauses = []
            if isinstance(el, dict):
                clauses = el.get("clauses") or []
            # Heuristic:
            # - "logic-builder" = buy_filter is none AND entry_logic has at least one non-empty clause
            # - "always"       = buy_filter none AND entry_logic clauses empty
            # - otherwise      = simple buy_filter mode
            if bf in {"none", ""}:
                if clauses and any(isinstance(c, list) and len(c) > 0 for c in clauses):
                    logic += 1
                    logic_clauses_sum += int(len(clauses))
                    logic_conds_sum += int(sum(len(c) for c in clauses if isinstance(c, list)))
                else:
                    always += 1
            else:
                filter_counts[bf] += 1

            try:
                if float(params.get("trail_pct", 0.0) or 0.0) > 0.0:
                    trail += 1
            except Exception:
                pass
            try:
                if int(params.get("max_hold_bars", 0) or 0) > 0:
                    time_stop += 1
            except Exception:
                pass
            try:
                if float(params.get("sl_pct", 0.0) or 0.0) > 0.0:
                    sl += 1
            except Exception:
                pass
            try:
                if float(params.get("tp_pct", 0.0) or 0.0) > 0.0:
                    tp += 1
            except Exception:
                pass

    comp: Dict[str, Any] = {
        "generated_at": _dt.utcnow().isoformat(),
        "total": int(total),
        "entry": {
            "logic_builder": int(logic),
            "always": int(always),
            "simple_filters": {k: int(v) for k, v in sorted(filter_counts.items(), key=lambda kv: (-kv[1], kv[0]))},
            "logic_avg_clauses": (logic_clauses_sum / logic) if logic else 0.0,
            "logic_avg_conditions": (logic_conds_sum / logic) if logic else 0.0,
        },
        "exits": {
            "trailing_stop_enabled": int(trail),
            "time_stop_enabled": int(time_stop),
            "stop_loss_enabled": int(sl),
            "take_profit_enabled": int(tp),
        },
    }
    return comp


def _render_grid_composition(comp: Dict[str, Any]) -> None:
    """Small UI block: what did we generate?"""
    total = int(comp.get("total", 0) or 0)
    if total <= 0:
        st.info("No grid composition available.")
        return

    entry = comp.get("entry") or {}
    exits = comp.get("exits") or {}

    def _pct(x: int) -> str:
        return f"{(100.0 * float(x) / float(total)):.0f}%"

    logic_n = int(entry.get("logic_builder", 0) or 0)
    always_n = int(entry.get("always", 0) or 0)
    trail_n = int(exits.get("trailing_stop_enabled", 0) or 0)
    time_n = int(exits.get("time_stop_enabled", 0) or 0)

    st.markdown("### Generated composition")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total variants", f"{total:,}")
    c2.metric("Logic-builder", f"{_pct(logic_n)}", f"{logic_n:,}")
    c3.metric("Trailing stop enabled", f"{_pct(trail_n)}", f"{trail_n:,}")
    c4.metric("Time stop enabled", f"{_pct(time_n)}", f"{time_n:,}")

    # Simple filter breakdown (only show if present)
    sf = entry.get("simple_filters") or {}
    if isinstance(sf, dict) and len(sf) > 0:
        nice = {
            "below_ema": "Dip below EMA",
            "rsi_below": "RSI low",
            "bb_z_below": "BB z (oversold)",
            "macd_bull": "MACD bullish",
            "adx_above": "ADX trending",
            "donch_pos_below": "Donchian bottom",
        }
        parts = []
        for k, v in sf.items():
            try:
                v = int(v)
            except Exception:
                continue
            parts.append(f"{nice.get(k, k)}: {v} ({_pct(v)})")
        if parts:
            st.caption("Simple filters: " + " · ".join(parts))

    # Always-buy share (explicit, because it matters for interpretability)
    st.caption(f"Always-buy (no entry filter): {always_n} ({_pct(always_n)})")

    # Logic complexity (just factual)
    avg_c = float(entry.get("logic_avg_clauses", 0.0) or 0.0)
    avg_k = float(entry.get("logic_avg_conditions", 0.0) or 0.0)
    if logic_n > 0:
        st.caption(f"Logic-builder complexity (avg): {avg_c:.2f} clauses · {avg_k:.2f} conditions")

def _tail_jsonl(path: Path, *, max_lines: int = 400) -> List[Dict[str, Any]]:
    """Read the last N JSONL rows (best-effort)."""
    if not path or (not path.exists()):
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        lines = lines[-int(max_lines) :]
        out: List[Dict[str, Any]] = []
        for s in lines:
            s = s.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
        return out
    except Exception:
        return []



def _render_run_monitor(progress_path: Optional[Path]) -> None:
    """Render run progress from a JSONL file *or* a directory of JSONL files.

    Some stages may write telemetry to different files (e.g., rerun.jsonl), so
    supporting a directory keeps the UI live without having to guess filenames.
    """

    if progress_path is None:
        st.info("Waiting for progress telemetry…")
        return

    paths: List[Path] = []
    if progress_path.is_dir():
        paths = sorted(progress_path.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not paths:
            st.info("Waiting for progress telemetry…")
            return
    else:
        if not progress_path.exists():
            st.info("Waiting for progress telemetry…")
            return
        paths = [progress_path]

    events: List[Dict[str, Any]] = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                # tag the source file (useful when debugging)
                obj.setdefault("_src", p.name)
                events.append(obj)

    if not events:
        st.info("Waiting for progress telemetry…")
        return

    # Sort by timestamp if available; otherwise preserve file order.
    def _t(e: Dict[str, Any]) -> float:
        t = e.get("t")
        try:
            return float(t)
        except Exception:
            return float("nan")

    # If at least some events have timestamps, sort by them.
    if any(isinstance(e.get("t"), (int, float)) for e in events):
        events = sorted(events, key=lambda e: (_t(e) if _t(e) == _t(e) else float("inf")))

    last = events[-1]
    df = pd.DataFrame(events)

    stage = str(last.get("stage", ""))
    # Friendlier labels (avoid dev-y stage:phase spam)
    _stage_key = stage.split(":")[0].strip().lower()
    _phase_key = str(last.get("phase", "")).split(":")[0].strip().lower()
    _stage_map = {
        "batch": "Batch sweep",
        "rolling_starts": "Rolling Starts",
        "walkforward": "Walkforward",
        "postprocess": "Postprocess",
    }
    _phase_map = {
        "run": "running",
        "artifacts": "finalizing",
        "done": "done",
        "rerun": "rerun",
        "rank": "ranking",
    }
    stage_disp = _stage_map.get(_stage_key, stage or "(unknown)")
    phase_disp = _phase_map.get(_phase_key, _phase_key) if _phase_key else ""
    phase = str(last.get("phase", ""))
    done = last.get("i", last.get("done", last.get("n_done", 0)))
    total = last.get("n", last.get("total", last.get("n_total", 0)))

    # Header metrics
    cols = st.columns(4)
    with cols[0]:
        st.metric("Stage", (f"{stage_disp} — {phase_disp}" if phase_disp else stage_disp) or "(unknown)")
    with cols[1]:
        try:
            st.metric("Progress", f"{int(done)}/{int(total)}")
        except Exception:
            st.metric("Progress", f"{done}/{total}")
    with cols[2]:
        rate = last.get("rate", last.get("throughput"))
        if rate is None:
            st.metric("Throughput", "")
        else:
            try:
                st.metric("Throughput", f"{float(rate):.2f}/s")
            except Exception:
                st.metric("Throughput", str(rate))
    with cols[3]:
        # Prefer showing survivors if available, otherwise show message
        surv = last.get("survivors")
        if isinstance(surv, int):
            st.metric("Survivors", str(surv))
        else:
            st.metric("Message", str(last.get("message", ""))[:32])

    # Progress bar
    try:
        done_f = float(done)
        total_f = float(total)
        if total_f > 0:
            st.progress(min(1.0, max(0.0, done_f / total_f)))
    except Exception:
        pass

    # Mini charts (rate + best-so-far)
    if "t" in df.columns:
        if "rate" in df.columns:
            s = pd.to_numeric(df["rate"], errors="coerce").dropna()
            if len(s) >= 3:
                st.line_chart(df.set_index("t")["rate"], height=140)

        # Best: support either a numeric 'best' column OR dict-shaped best_detail
        best_series = None
        if "best" in df.columns:
            b = pd.to_numeric(df["best"], errors="coerce")
            if b.notna().sum() >= 3:
                best_series = b

        if best_series is None and "best_detail" in df.columns:
            def _best_from_detail(x: Any) -> float:
                if isinstance(x, dict):
                    # common shapes
                    if "value" in x:
                        try:
                            return float(x.get("value"))
                        except Exception:
                            return float("nan")
                    # pick median_window_return if present, else first numeric
                    if "median_window_return" in x:
                        try:
                            return float(x.get("median_window_return"))
                        except Exception:
                            return float("nan")
                    for v in x.values():
                        try:
                            fv = float(v)
                            if math.isfinite(fv):
                                return fv
                        except Exception:
                            continue
                return float("nan")
            b2 = df["best_detail"].apply(_best_from_detail)
            if pd.to_numeric(b2, errors="coerce").notna().sum() >= 3:
                best_series = pd.to_numeric(b2, errors="coerce")

        if best_series is not None:
            st.line_chart(pd.DataFrame({"best": best_series, "t": df["t"]}).set_index("t")["best"], height=140)

        # Optional: best_pct if present
        if "best_pct" in df.columns:
            p = pd.to_numeric(df["best_pct"], errors="coerce").dropna()
            if len(p) >= 3:
                st.line_chart(df.set_index("t")["best_pct"], height=140)

    # Failure breakdown if present (support 'fails' dict or 'fail_top' list)
    fails = last.get("fails")
    if not (isinstance(fails, dict) and fails):
        ft = last.get("fail_top")
        if isinstance(ft, list) and ft:
            try:
                fails = {str(k): int(v) for k, v in ft}
            except Exception:
                fails = None

    if isinstance(fails, dict) and fails:
        st.caption("Gate rejects (so far)")
        s = pd.Series({str(k): int(v) for k, v in fails.items()}).sort_values(ascending=False)
        st.bar_chart(s, height=160)



def _tail_text(lines: Iterable[str], n: int = 40) -> str:
    xs = list(lines)[-max(0, int(n)) :]
    return "".join(xs)


def _run_cmd(
    cmd: List[str],
    *,
    cwd: Path,
    label: str,
    progress_path: Optional[Path] = None,
    refresh_hz: float = 4.0,
) -> None:
    """Run a command and stream output + telemetry into the UI.

    Sprint 3 polish:
    - Logs are hidden by default (toggle Debug in the sidebar)
    - On failure, show a short summary + last output lines
    """
    if not cmd:
        raise ValueError("Empty command")

    # Make "python" consistent across platforms
    if str(cmd[0]).lower() in {"python", "py", "py.exe", "python3"}:
        cmd = [PY, *cmd[1:]]

    debug = bool(st.session_state.get("ui.debug", False))

    with st.expander(label, expanded=True):
        details = st.expander("Details (command + logs)", expanded=debug)
        with details:
            st.code(" ".join(cmd), language="bash")
            if not debug:
                st.caption("Debug is off — raw logs are hidden. Toggle Debug in the sidebar to view streaming output.")
            log_ph = st.empty()

        mon_ph = st.empty()

        # Spawn process
        t0 = time.time()
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        q: "queue.Queue[str]" = queue.Queue()

        def _reader():
            try:
                assert p.stdout is not None
                for line in p.stdout:
                    q.put(line)
            except Exception:
                pass

        th = threading.Thread(target=_reader, daemon=True)
        th.start()

        lines = deque(maxlen=800)
        sleep_s = max(0.05, 1.0 / float(max(1.0, refresh_hz)))

        while p.poll() is None:
            # Drain output queue
            for _ in range(400):
                try:
                    lines.append(q.get_nowait())
                except Exception:
                    break

            with mon_ph.container():
                _render_run_monitor(progress_path)

            if debug and lines:
                log_ph.code("".join(list(lines)[-140:]), language="text")

            time.sleep(sleep_s)

        # Final drain
        for _ in range(8000):
            try:
                lines.append(q.get_nowait())
            except Exception:
                break

        dt = time.time() - t0
        rc = int(p.returncode or 0)

        # Always show logs on failure (even if debug is off)
        if rc != 0:
            tail = _tail_text(lines, n=60)
            with details:
                st.error(f"Failed (code={rc}) after {dt:.1f}s")
                st.code(tail or "(no output captured)", language="text")
            raise RuntimeError(f"{label} failed (code={rc}) after {dt:.1f}s")

        # On success, only show full logs in debug mode
        if debug and lines:
            with details:
                st.caption(f"Completed in {dt:.1f}s")
                st.code("".join(lines), language="text")


def _run_subprocess_stream(
    cmd: List[str],
    *,
    cwd: Path,
    label: str,
    ui_ph: "st.delta_generator.DeltaGenerator",
    progress_path: Optional[Path] = None,
    refresh_hz: float = 4.0,
    debug: Optional[bool] = None,
) -> Tuple[int, float, str]:
    """Run a command and stream progress/logs into a *single* UI panel.

    Returns: (returncode, seconds, tail_text)
    """
    if not cmd:
        raise ValueError("Empty command")

    if str(cmd[0]).lower() in {"python", "py", "py.exe", "python3"}:
        cmd = [PY, *cmd[1:]]

    if debug is None:
        debug = bool(st.session_state.get("ui.debug", False))

    with ui_ph.container():
        st.write(f"### {label}")
        mon_ph = st.empty()
        details = st.expander("Details (command + logs)", expanded=bool(debug))
        with details:
            st.code(" ".join(cmd), language="bash")
            if not debug:
                st.caption("Debug is off — streaming logs are hidden. Toggle Debug in the sidebar to view them.")
            log_ph = st.empty()

    # Spawn process
    t0 = time.time()
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    q: "queue.Queue[str]" = queue.Queue()

    def _reader():
        try:
            assert p.stdout is not None
            for line in p.stdout:
                q.put(line)
        except Exception:
            pass

    th = threading.Thread(target=_reader, daemon=True)
    th.start()

    lines = deque(maxlen=800)
    sleep_s = max(0.05, 1.0 / float(max(1.0, refresh_hz)))

    while p.poll() is None:
        for _ in range(400):
            try:
                lines.append(q.get_nowait())
            except Exception:
                break

        with mon_ph.container():
            if progress_path is not None:
                _render_run_monitor(progress_path)
            else:
                st.info("Running…")

        if bool(debug) and lines:
            with details:
                log_ph.code("".join(list(lines)[-140:]), language="text")

        time.sleep(sleep_s)

    # Final drain
    for _ in range(8000):
        try:
            lines.append(q.get_nowait())
        except Exception:
            break

    dt = time.time() - t0
    rc = int(p.returncode or 0)
    tail = _tail_text(lines, n=80)

    if rc != 0:
        with ui_ph.container():
            st.error(f"{label} failed (code={rc}) after {dt:.1f}s")
            with details:
                st.code(tail or "(no output captured)", language="text")

    return rc, dt, tail


@dataclass
class _PipelineStage:
    key: str
    label: str


class _PipelineUI:
    """A tiny 'lab monitor' that runs stages sequentially and keeps the UI clean.

    - One stepper strip
    - One active-stage panel (only one stage expanded at a time)
    - Logs hidden by default (Debug toggle)
    """

    def __init__(self, stages: List[_PipelineStage], *, debug: Optional[bool] = None):
        self.stages = stages
        self.debug = bool(st.session_state.get("ui.debug", False)) if debug is None else bool(debug)
        self.status: Dict[str, str] = {s.key: "pending" for s in stages}  # pending|running|done|fail
        self.durations: Dict[str, float] = {}
        self.stepper_ph = st.empty()
        self.active_ph = st.empty()
        self.render()

    def _icon(self, stt: str) -> str:
        return {
            "pending": "⬜",
            "running": "⏳",
            "done": "✅",
            "fail": "❌",
        }.get(stt, "⬜")

    def render(self) -> None:
        with self.stepper_ph.container():
            cols = st.columns(len(self.stages))
            for col, s in zip(cols, self.stages):
                stt = self.status.get(s.key, "pending")
                icon = self._icon(stt)
                dur = self.durations.get(s.key)
                col.markdown(f"{icon} **{s.label}**")
                if dur is not None:
                    col.caption(f"{dur:.1f}s")

    def run(self, key: str, cmd: List[str], *, cwd: Path, progress_path: Optional[Path] = None) -> None:
        label = next((s.label for s in self.stages if s.key == key), key)
        self.status[key] = "running"
        self.render()

        rc, dt, tail = _run_subprocess_stream(
            cmd,
            cwd=cwd,
            label=label,
            ui_ph=self.active_ph,
            progress_path=progress_path,
            debug=self.debug,
        )
        self.durations[key] = float(dt)

        if rc != 0:
            self.status[key] = "fail"
            self.render()
            raise RuntimeError(f"{label} failed (code={rc})")

        self.status[key] = "done"
        self.render()

def _list_runs() -> List[Path]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    xs = [p for p in RUNS_DIR.glob("batch_*") if p.is_dir()]
    return sorted(xs, key=lambda p: p.stat().st_mtime, reverse=True)

def _has_any_results(run_dir: Path) -> bool:
    """True if the run folder contains any batch result CSV."""
    for fn in (
        "results_full_passed.csv",
        "results_passed.csv",
        "results_full.csv",
        "results.csv",
    ):
        if (run_dir / fn).exists():
            return True
    return False



def _pick_latest_dir(base: Path, glob_pat: str) -> Optional[Path]:
    if not base.exists():
        return None
    xs = [p for p in base.glob(glob_pat) if p.is_dir()]
    if not xs:
        return None
    return sorted(xs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _parse_top_artifact_dirs(run_dir: Path) -> Dict[str, Path]:
    """
    Map config_id -> top artifact dir, when present.
    Folder format: top/0001_<config_id>_<label>/
    """
    top_dir = run_dir / "top"
    out: Dict[str, Path] = {}
    if not top_dir.exists():
        return out
    for d in top_dir.iterdir():
        if not d.is_dir():
            continue
        parts = d.name.split("_", 2)
        if len(parts) < 2:
            continue
        cid = parts[1]
        out[str(cid)] = d
    return out


# =============================================================================
# "Questions" (stage filters)
# =============================================================================

_SEVERITY_ORDER = {"info": 0, "warn": 1, "warning": 1, "critical": 2, "crit": 2}


@dataclass(frozen=True)
class ConstraintSpec:
    metric_id: str
    op: str  # ">=" or "<="
    threshold: float
    severity: str  # "info" | "warn" | "critical"
    note: str = ""


@dataclass(frozen=True)
class ChoiceSpec:
    label: str
    constraints: List[ConstraintSpec]


@dataclass(frozen=True)
class QuestionSpec:
    id: str
    title: str
    explanation: str
    choices: List[ChoiceSpec]
    default_index: int = 0


@dataclass
class EvalOutcome:
    verdict: str  # PASS | WARN | FAIL
    crits: int
    warns: int
    infos: int
    missing: int
    violations: List[Dict[str, Any]]
    missing_metrics: List[str]


def _to_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _passes(op: str, value: float, thr: float) -> bool:
    if op == ">=":
        return value >= thr
    if op == "<=":
        return value <= thr
    return True


def evaluate_row_with_questions(
    row: Dict[str, Any],
    questions: List[QuestionSpec],
    answers: Dict[str, int],
) -> EvalOutcome:
    violations: List[Dict[str, Any]] = []
    missing_metrics: List[str] = []

    crits = warns = infos = 0
    missing = 0

    for q in questions:
        pick = int(answers.get(q.id, q.default_index))
        pick = max(0, min(pick, len(q.choices) - 1))
        choice = q.choices[pick]

        for c in choice.constraints:
            v = _to_float(row.get(c.metric_id, float("nan")))
            if v != v:  # NaN
                missing += 1
                missing_metrics.append(c.metric_id)
                continue

            if _passes(c.op, v, float(c.threshold)):
                continue

            sev = str(c.severity).strip().lower()
            sev_rank = _SEVERITY_ORDER.get(sev, 0)

            if sev_rank >= 2:
                crits += 1
            elif sev_rank == 1:
                warns += 1
            else:
                infos += 1

            violations.append(
                {
                    "question_id": q.id,
                    "question": q.title,
                    "metric_id": c.metric_id,
                    "metric": _metric_label(c.metric_id),
                    "value": v,
                    "op": c.op,
                    "threshold": float(c.threshold),
                    "severity": sev,
                    "message": f"{_metric_label(c.metric_id)} is {_metric_fmt(c.metric_id, v)} but your limit is {c.op} {_metric_fmt(c.metric_id, float(c.threshold))}.",
                    "note": c.note,
                }
            )

    verdict = "PASS"
    if crits > 0:
        verdict = "FAIL"
    elif warns > 0:
        verdict = "WARN"

    # Make missing list stable unique
    missing_metrics2 = sorted({str(x) for x in missing_metrics})

    return EvalOutcome(
        verdict=verdict,
        crits=int(crits),
        warns=int(warns),
        infos=int(infos),
        missing=int(missing),
        violations=violations,
        missing_metrics=missing_metrics2,
    )


def apply_stage_eval(
    df: pd.DataFrame,
    *,
    stage_key: str,
    questions: List[QuestionSpec],
    answers: Dict[str, int],
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    rows = df.to_dict(orient="records")
    verdicts: List[str] = []
    crits: List[int] = []
    warns: List[int] = []
    missing: List[int] = []

    for r in rows:
        out = evaluate_row_with_questions(r, questions, answers)
        verdicts.append(out.verdict)
        crits.append(out.crits)
        warns.append(out.warns)
        missing.append(out.missing)

    out_df = df.copy()
    out_df[f"{stage_key}.verdict"] = verdicts
    out_df[f"{stage_key}.crits"] = crits
    out_df[f"{stage_key}.warns"] = warns
    out_df[f"{stage_key}.missing"] = missing
    return out_df


def _question_ui(questions: List[QuestionSpec], *, key_prefix: str) -> Dict[str, int]:
    answers: Dict[str, int] = {}
    for q in questions:
        opts = [c.label for c in q.choices]
        idx = st.radio(
            q.title,
            options=list(range(len(opts))),
            format_func=lambda i: opts[int(i)],
            index=int(q.default_index),
            key=f"{key_prefix}.{q.id}",
        )
        st.caption(q.explanation)
        answers[q.id] = int(idx)
    return answers


def batch_questions() -> List[QuestionSpec]:
    return [
        QuestionSpec(
            id="batch_drawdown",
            title="How big of a drop from a previous high can you tolerate?",
            explanation="Single-run max drawdown on equity curve.",
            choices=[
                ChoiceSpec("Max 20% drop", [ConstraintSpec("performance.max_drawdown_equity", "<=", 0.20, "critical")]),
                ChoiceSpec("Max 35% drop", [ConstraintSpec("performance.max_drawdown_equity", "<=", 0.35, "warn")]),
                ChoiceSpec("Max 50% drop", [ConstraintSpec("performance.max_drawdown_equity", "<=", 0.50, "warn")]),
                ChoiceSpec("Max 70% drop", [ConstraintSpec("performance.max_drawdown_equity", "<=", 0.70, "info")]),
                ChoiceSpec("Don't filter on this", []),
            ],
            default_index=1,
        ),
        QuestionSpec(
            id="batch_profit",
            title="Do you require net profit (excluding deposits) to be positive?",
            explanation="If deposits are part of the plan, this isolates whether the strategy actually made money beyond what you put in.",
            choices=[
                ChoiceSpec("Yes (must be ≥ $0)", [ConstraintSpec("equity.net_profit_ex_cashflows", ">=", 0.0, "warn")]),
                ChoiceSpec("No (let losers through)", []),
            ],
            default_index=0,
        ),
        QuestionSpec(
            id="batch_fees",
            title="How sensitive are you to fees and churn?",
            explanation="Higher turnover usually means more slippage/fees in real life.",
            choices=[
                ChoiceSpec(
                    "Very fee-sensitive",
                    [
                        ConstraintSpec("exposure.turnover_notional_over_avg_equity", "<=", 0.5, "warn"),
                        ConstraintSpec("efficiency.fee_impact_pct", "<=", 10.0, "warn"),
                    ],
                ),
                ChoiceSpec(
                    "Moderately fee-sensitive",
                    [
                        ConstraintSpec("exposure.turnover_notional_over_avg_equity", "<=", 1.5, "warn"),
                        ConstraintSpec("efficiency.fee_impact_pct", "<=", 25.0, "warn"),
                    ],
                ),
                ChoiceSpec("Not very fee-sensitive", [ConstraintSpec("exposure.turnover_notional_over_avg_equity", "<=", 3.0, "info")]),
                ChoiceSpec("Don't filter on this", []),
            ],
            default_index=1,
        ),
    ]


def rolling_questions() -> List[QuestionSpec]:
    return [
        QuestionSpec(
            id="rs_worst_return",
            title="If you started at a bad time, how much loss can you tolerate?",
            explanation="We simulate many start dates. This checks the worst 10% outcome (p10) time-weighted return.",
            choices=[
                ChoiceSpec("Worst 10% must not lose money", [ConstraintSpec("twr_p10", ">=", 0.0, "critical")]),
                ChoiceSpec("I can tolerate up to -10%", [ConstraintSpec("twr_p10", ">=", -0.10, "warn")]),
                ChoiceSpec("I can tolerate up to -25%", [ConstraintSpec("twr_p10", ">=", -0.25, "warn")]),
                ChoiceSpec("I can tolerate up to -50%", [ConstraintSpec("twr_p10", ">=", -0.50, "info")]),
                ChoiceSpec("Don't filter on this", []),
            ],
            default_index=1,
        ),
        QuestionSpec(
            id="rs_drawdown",
            title="In a bad-but-common scenario, how deep a drawdown can you tolerate?",
            explanation="This uses the p90 drawdown across rolling starts.",
            choices=[
                ChoiceSpec("Max 20% drop", [ConstraintSpec("dd_p90", "<=", 0.20, "critical")]),
                ChoiceSpec("Max 35% drop", [ConstraintSpec("dd_p90", "<=", 0.35, "warn")]),
                ChoiceSpec("Max 50% drop", [ConstraintSpec("dd_p90", "<=", 0.50, "warn")]),
                ChoiceSpec("Max 70% drop", [ConstraintSpec("dd_p90", "<=", 0.70, "info")]),
                ChoiceSpec("Don't filter on this", []),
            ],
            default_index=1,
        ),
        QuestionSpec(
            id="rs_underwater",
            title="How long can you tolerate being underwater (below a prior high)?",
            explanation="This uses p90 underwater duration (days) across rolling starts.",
            choices=[
                ChoiceSpec("About 1 month", [ConstraintSpec("uw_p90_days", "<=", 30.0, "critical")]),
                ChoiceSpec("About 3 months", [ConstraintSpec("uw_p90_days", "<=", 90.0, "warn")]),
                ChoiceSpec("About 6 months", [ConstraintSpec("uw_p90_days", "<=", 180.0, "warn")]),
                ChoiceSpec("About 1 year", [ConstraintSpec("uw_p90_days", "<=", 365.0, "info")]),
                ChoiceSpec("Don't filter on this", []),
            ],
            default_index=2,
        ),
        QuestionSpec(
            id="rs_util",
            title="Do you want this plan invested most of the time?",
            explanation="Median invested fraction across rolling starts.",
            choices=[
                ChoiceSpec("Mostly invested", [ConstraintSpec("util_p50", ">=", 0.75, "warn")]),
                ChoiceSpec("Balanced", [ConstraintSpec("util_p50", ">=", 0.50, "info")]),
                ChoiceSpec("Mostly in cash is fine", [ConstraintSpec("util_p50", ">=", 0.25, "info")]),
                ChoiceSpec("I don't care", []),
            ],
            default_index=1,
        ),
    ]


def walkforward_questions() -> List[QuestionSpec]:
    # Walkforward metrics are produced by engine.walkforward -> wf_summary.csv
    #
    # Philosophy: WF exists to punish "in-sample optimism".
    # Defaults below are intentionally mild (WARN/INFO) so users can explore,
    # but the questions steer attention toward worst-case and stability.
    return [
        QuestionSpec(
            id="wf_typical",
            title="Do you require typical walk-forward performance to be positive?",
            explanation="Median (p50) return across walk-forward windows.",
            choices=[
                ChoiceSpec("Yes (p50 ≥ 0)", [ConstraintSpec("return_p50", ">=", 0.0, "warn")]),
                ChoiceSpec("No", []),
            ],
            default_index=0,
        ),
        QuestionSpec(
            id="wf_worst_typical",
            title="How negative can the 'worst typical' window be?",
            explanation="10th percentile (p10) return across windows. This is a good 'robustness' anchor.",
            choices=[
                ChoiceSpec("p10 ≥ -5%", [ConstraintSpec("return_p10", ">=", -0.05, "warn")]),
                ChoiceSpec("p10 ≥ -10%", [ConstraintSpec("return_p10", ">=", -0.10, "info")]),
                ChoiceSpec("Don't filter on this", []),
            ],
            default_index=1,
        ),
        QuestionSpec(
            id="wf_min",
            title="How bad can the single worst window be?",
            explanation="Minimum window return across all walk-forward windows (the absolute faceplant).",
            choices=[
                ChoiceSpec("Worst ≥ -10%", [ConstraintSpec("min_window_return", ">=", -0.10, "warn")]),
                ChoiceSpec("Worst ≥ -25%", [ConstraintSpec("min_window_return", ">=", -0.25, "info")]),
                ChoiceSpec("Don't filter on this", []),
            ],
            default_index=1,
        ),
        QuestionSpec(
            id="wf_dd",
            title="How much drawdown pain can you tolerate in walk-forward?",
            explanation="90th percentile max drawdown (dd_p90) across windows. Lower is better.",
            choices=[
                ChoiceSpec("dd_p90 ≤ 20%", [ConstraintSpec("dd_p90", "<=", 0.20, "warn")]),
                ChoiceSpec("dd_p90 ≤ 35%", [ConstraintSpec("dd_p90", "<=", 0.35, "info")]),
                ChoiceSpec("Don't filter on this", []),
            ],
            default_index=1,
        ),
        QuestionSpec(
            id="wf_consistency",
            title="How consistent should it be across windows?",
            explanation="Percent of windows with positive return.",
            choices=[
                ChoiceSpec("≥ 60% profitable windows", [ConstraintSpec("pct_profitable_windows", ">=", 0.60, "warn")]),
                ChoiceSpec("≥ 50% profitable windows", [ConstraintSpec("pct_profitable_windows", ">=", 0.50, "info")]),
                ChoiceSpec("Don't filter on this", []),
            ],
            default_index=1,
        ),
        QuestionSpec(
            id="wf_trading",
            title="Should it actually trade in most windows?",
            explanation="Percent of windows with at least 1 trade. Avoids strategies that only 'wake up' rarely.",
            choices=[
                ChoiceSpec("≥ 80% windows traded", [ConstraintSpec("pct_windows_traded", ">=", 0.80, "warn")]),
                ChoiceSpec("≥ 60% windows traded", [ConstraintSpec("pct_windows_traded", ">=", 0.60, "info")]),
                ChoiceSpec("Don't filter on this", []),
            ],
            default_index=1,
        ),
    ]


# =============================================================================
# Run data loaders / mergers
# =============================================================================

def _ensure_config_id(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a canonical string config_id column.

    We strongly prefer the flattened column name from the batch runner (config.id),
    because some artifacts may include a legacy/config_id column that is not stable
    across stages. This keeps RS/WF joins and evidence lookups consistent.
    """
    if df is None or df.empty:
        return df
    out = df.copy()

    if "config.id" in out.columns:
        out["config_id"] = out["config.id"].astype(str).str.strip()
    elif "config_id" in out.columns:
        out["config_id"] = out["config_id"].astype(str).str.strip()
    else:
        # last-resort fallbacks (rare)
        for alt in ["config.id", "config_id", "id", "cfg_id"]:
            if alt in out.columns:
                out["config_id"] = out[alt].astype(str).str.strip()
                break

    if "config.label" in out.columns:
        out["config.label"] = out["config.label"].astype(str)
    return out



def load_batch_frames(run_dir: Path) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Returns:
      - sweep_all: results.csv
      - sweep_passed: results_passed.csv
      - full_all: results_full.csv
      - full_passed: results_full_passed.csv
      - ranked: post/ranked.csv
    """
    frames: Dict[str, Optional[pd.DataFrame]] = {}
    frames["sweep_all"] = _load_csv(run_dir / "results.csv")
    frames["sweep_passed"] = _load_csv(run_dir / "results_passed.csv")
    frames["full_all"] = _load_csv(run_dir / "results_full.csv")
    frames["full_passed"] = _load_csv(run_dir / "results_full_passed.csv")
    frames["ranked"] = _load_csv(run_dir / "post" / "ranked.csv")

    for k, v in list(frames.items()):
        if v is not None:
            frames[k] = _ensure_config_id(v)

    return frames


def pick_survivors(frames: Dict[str, Optional[pd.DataFrame]]) -> Tuple[pd.DataFrame, str]:
    """
    Pick the 'survivor' set we treat as the batch output.
    Preference order:
      1) full_passed (rerun-passed configs) if exists and non-empty
      2) sweep_passed (sweep-passed configs)
      3) full_all
      4) sweep_all
    """
    for key in ["full_passed", "sweep_passed", "full_all", "sweep_all"]:
        df = frames.get(key)
        if df is not None and not df.empty:
            return df.copy(), key
    return pd.DataFrame([]), "none"


def load_rs_summary(run_dir: Path, rs_dir: Optional[Path]) -> Optional[pd.DataFrame]:
    if rs_dir is None:
        return None
    p = rs_dir / "rolling_starts_summary.csv"
    df = _load_csv(p)
    if df is None or df.empty:
        return df
    df = df.copy()
    if "config_id" in df.columns:
        df["config_id"] = df["config_id"].astype(str).str.strip()
    return df


def load_rs_detail(run_dir: Path, rs_dir: Optional[Path]) -> Optional[pd.DataFrame]:
    if rs_dir is None:
        return None
    p = rs_dir / "rolling_starts_detail.csv"
    df = _load_csv(p)
    if df is None or df.empty:
        return df
    df = df.copy()
    if "config_id" in df.columns:
        df["config_id"] = df["config_id"].astype(str).str.strip()
    return df


def load_wf_summary(wf_dir: Optional[Path]) -> Optional[pd.DataFrame]:
    if wf_dir is None:
        return None
    p = wf_dir / "wf_summary.csv"
    df = _load_csv(p)
    if df is None or df.empty:
        return df
    df = df.copy()
    if "config_id" in df.columns:
        df["config_id"] = df["config_id"].astype(str).str.strip()
    # normalize pct column name for our questions (keep original too)
    if "pct_profitable_windows" in df.columns:
        df["pct_profitable_windows"] = pd.to_numeric(df["pct_profitable_windows"], errors="coerce")
    return df


def load_wf_results(wf_dir: Optional[Path]) -> Optional[pd.DataFrame]:
    if wf_dir is None:
        return None
    p = wf_dir / "wf_results.csv"
    df = _load_csv(p)
    if df is None or df.empty:
        return df
    df = df.copy()
    if "config_id" in df.columns:
        df["config_id"] = df["config_id"].astype(str).str.strip()
    return df


def merge_stage(
    base: pd.DataFrame,
    add: Optional[pd.DataFrame],
    *,
    on: str = "config_id",
    suffix: str = "",
) -> pd.DataFrame:
    if base is None or base.empty:
        return base
    if add is None or add.empty or on not in add.columns:
        out = base.copy()
        out[f"{suffix}.measured" if suffix else "measured"] = False
        return out

    out = base.merge(add, how="left", on=on, suffixes=("", f".{suffix}"))
    out[f"{suffix}.measured" if suffix else "measured"] = out[on].isin(add[on].astype(str))
    return out


# =============================================================================
# New run wizard: DCA/Swing baseline builder
# =============================================================================

def _filter_true_pct(
    df_feat: pd.DataFrame,
    buy_filter: str,
    *,
    ema_len: int = 200,
    rsi_thr: float = 40.0,
    macd_hist_thr: float = 0.0,
    bb_z_thr: float = -1.0,
    adx_thr: float = 20.0,
    donch_pos_thr: float = 0.20,
) -> Optional[float]:
    """Percent of bars where the entry filter is true (for Build-step context)."""
    if df_feat is None or df_feat.empty:
        return None
    f = str(buy_filter or "none").strip().lower()

    try:
        close = df_feat["close"].astype(float)
    except Exception:
        return None

    mask = None
    if f in {"none", ""}:
        mask = pd.Series(True, index=df_feat.index)
    elif f == "below_ema":
        col = f"ema_{int(ema_len)}"
        if col in df_feat.columns:
            mask = close <= pd.to_numeric(df_feat[col], errors="coerce")
    elif f == "rsi_below":
        col = "rsi_14"
        if col in df_feat.columns:
            mask = pd.to_numeric(df_feat[col], errors="coerce") <= float(rsi_thr)
    elif f == "macd_bull":
        col = "macd_hist_12_26_9"
        if col in df_feat.columns:
            mask = pd.to_numeric(df_feat[col], errors="coerce") >= float(macd_hist_thr)
    elif f == "bb_z_below":
        col = "bb_z_20"
        if col in df_feat.columns:
            mask = pd.to_numeric(df_feat[col], errors="coerce") <= float(bb_z_thr)
    elif f == "adx_above":
        col = "adx_14"
        if col in df_feat.columns:
            mask = pd.to_numeric(df_feat[col], errors="coerce") >= float(adx_thr)
    elif f == "donch_pos_below":
        col = "donch_pos_20"
        if col in df_feat.columns:
            mask = pd.to_numeric(df_feat[col], errors="coerce") <= float(donch_pos_thr)

    if mask is None:
        return None
    try:
        return float(mask.mean() * 100.0)
    except Exception:
        return None



def _simple_filter_to_entry_logic(
    buy_filter: str,
    *,
    ema_len: int = 200,
    rsi_thr: float = 40.0,
    macd_hist_thr: float = 0.0,
    bb_z_thr: float = -1.0,
    adx_thr: float = 20.0,
    donch_pos_thr: float = 0.20,
) -> Dict[str, Any]:
    """
    Translate legacy single buy_filter into entry_logic (regime=[], clauses=[...]).
    entry_logic condition schema matches dca_swing_strategy_overhaul_v1.py:
      - indicator (lhs)
      - operator
      - threshold (rhs literal OR offset when ref_indicator used)
      - ref_indicator (optional rhs indicator)
    """
    f = str(buy_filter or "none").strip().lower()
    regime: List[Dict[str, Any]] = []
    clauses: List[List[Dict[str, Any]]] = []

    if f in {"none", ""}:
        return {"regime": regime, "clauses": clauses}  # no triggers => always allowed

    if f == "below_ema":
        clauses = [[{"indicator": "close", "operator": "<=", "ref_indicator": f"ema_{int(ema_len)}", "threshold": 0.0}]]
    elif f == "rsi_below":
        clauses = [[{"indicator": "rsi_14", "operator": "<=", "threshold": float(rsi_thr)}]]
    elif f == "bb_z_below":
        clauses = [[{"indicator": "bb_z_20", "operator": "<=", "threshold": float(bb_z_thr)}]]
    elif f == "macd_bull":
        clauses = [[{"indicator": "macd_hist_12_26_9", "operator": ">=", "threshold": float(macd_hist_thr)}]]
    elif f == "adx_above":
        clauses = [[{"indicator": "adx_14", "operator": ">=", "threshold": float(adx_thr)}]]
    elif f == "donch_pos_below":
        clauses = [[{"indicator": "donch_pos_20", "operator": "<=", "threshold": float(donch_pos_thr)}]]

    return {"regime": regime, "clauses": clauses}


def _cond_mask(df_feat: pd.DataFrame, cond: Dict[str, Any]) -> Optional[pd.Series]:
    """Vectorized mask for one condition against df_feat. Returns None if missing columns."""
    if df_feat is None or df_feat.empty or not isinstance(cond, dict):
        return None

    ind = cond.get("indicator") or cond.get("feature") or cond.get("lhs")
    op = str(cond.get("operator") or cond.get("op") or "").strip()
    thr = cond.get("threshold", cond.get("value", 0.0))
    ref = cond.get("ref_indicator") or cond.get("rhs") or cond.get("rhs_indicator")

    if op not in {"<", "<=", ">", ">="}:
        return None

    name = str(ind or "").strip()
    if not name:
        return None

    # lhs
    if name.lower() in {"open", "high", "low", "close", "volume", "vol"}:
        if name.lower() == "open":
            lhs = pd.to_numeric(df_feat.get("open"), errors="coerce")
        elif name.lower() == "high":
            lhs = pd.to_numeric(df_feat.get("high"), errors="coerce")
        elif name.lower() == "low":
            lhs = pd.to_numeric(df_feat.get("low"), errors="coerce")
        elif name.lower() == "close":
            lhs = pd.to_numeric(df_feat.get("close"), errors="coerce")
        else:
            lhs = pd.to_numeric(df_feat.get("volume"), errors="coerce")
    else:
        if name not in df_feat.columns:
            return None
        lhs = pd.to_numeric(df_feat[name], errors="coerce")

    # rhs: literal OR ref indicator (+ offset)
    if ref is not None and str(ref).strip():
        r = str(ref).strip()
        if r not in df_feat.columns and r.lower() not in {"open", "high", "low", "close", "volume", "vol"}:
            return None
        if r.lower() == "open":
            rhs0 = pd.to_numeric(df_feat.get("open"), errors="coerce")
        elif r.lower() == "high":
            rhs0 = pd.to_numeric(df_feat.get("high"), errors="coerce")
        elif r.lower() == "low":
            rhs0 = pd.to_numeric(df_feat.get("low"), errors="coerce")
        elif r.lower() == "close":
            rhs0 = pd.to_numeric(df_feat.get("close"), errors="coerce")
        elif r.lower() in {"volume", "vol"}:
            rhs0 = pd.to_numeric(df_feat.get("volume"), errors="coerce")
        else:
            rhs0 = pd.to_numeric(df_feat.get(r), errors="coerce")
        off = float(thr or 0.0)
        rhs = rhs0 + off
    else:
        rhs = float(thr)

    if op == "<":
        m = lhs < rhs
    elif op == "<=":
        m = lhs <= rhs
    elif op == ">":
        m = lhs > rhs
    else:
        m = lhs >= rhs

    return m.fillna(False)


def _entry_logic_masks(df_feat: pd.DataFrame, entry_logic: Dict[str, Any]) -> Optional[Tuple[pd.Series, pd.Series, pd.Series]]:
    """Return (regime_mask, entry_mask, combined_mask)."""
    if df_feat is None or df_feat.empty or not isinstance(entry_logic, dict):
        return None

    idx = df_feat.index
    regime_mask = pd.Series(True, index=idx)
    for c in entry_logic.get("regime", []) or []:
        cm = _cond_mask(df_feat, c)
        if cm is None:
            return None
        regime_mask &= cm

    clauses = entry_logic.get("clauses", []) or []
    if not clauses:
        entry_mask = pd.Series(True, index=idx)
    else:
        entry_mask = pd.Series(False, index=idx)
        for clause in clauses:
            if not clause:
                entry_mask |= True
                continue
            cm_all = pd.Series(True, index=idx)
            for c in clause:
                cm = _cond_mask(df_feat, c)
                if cm is None:
                    return None
                cm_all &= cm
            entry_mask |= cm_all

    combined = regime_mask & entry_mask
    return regime_mask, entry_mask, combined


def _entry_logic_true_pcts(df_feat: pd.DataFrame, entry_logic: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    """(regime%, entry%, combined%)"""
    masks = _entry_logic_masks(df_feat, entry_logic)
    if masks is None:
        return None
    r, e, c = masks
    return float(r.mean() * 100.0), float(e.mean() * 100.0), float(c.mean() * 100.0)


def _human_condition(cond: Dict[str, Any]) -> str:
    if not isinstance(cond, dict):
        return ""
    ind = cond.get("indicator") or cond.get("feature") or cond.get("lhs")
    op = cond.get("operator") or cond.get("op")
    thr = cond.get("threshold", cond.get("value", 0.0))
    ref = cond.get("ref_indicator") or cond.get("rhs") or cond.get("rhs_indicator")

    ind = str(ind or "").strip()
    op = str(op or "").strip()
    if ref is not None and str(ref).strip():
        r = str(ref).strip()
        off = float(thr or 0.0)
        if abs(off) < 1e-12:
            return f"{ind} {op} {r}"
        sign = "+" if off >= 0 else "-"
        return f"{ind} {op} {r} {sign} {abs(off):g}"
    else:
        try:
            v = float(thr)
            # prettier ints
            if abs(v - round(v)) < 1e-9:
                return f"{ind} {op} {int(round(v))}"
            return f"{ind} {op} {v:g}"
        except Exception:
            return f"{ind} {op} {thr}"


def _human_entry_logic(entry_logic: Dict[str, Any]) -> str:
    if not isinstance(entry_logic, dict):
        return ""
    parts: List[str] = []
    regime = entry_logic.get("regime", []) or []
    clauses = entry_logic.get("clauses", []) or []

    if regime:
        parts.append("Regime: " + " AND ".join([_human_condition(c) for c in regime if isinstance(c, dict)]))

    if not clauses:
        parts.append("Entry: (always allowed)")
    else:
        clause_strs = []
        for cl in clauses:
            if not cl:
                clause_strs.append("(always)")
            else:
                clause_strs.append("(" + " AND ".join([_human_condition(c) for c in cl if isinstance(c, dict)]) + ")")
        parts.append("Entry: " + " OR ".join(clause_strs))

    return " · ".join([p for p in parts if p.strip()])



def build_dca_baseline_params() -> Dict[str, Any]:
    st.subheader("Baseline plan (DCA/Swing)")

    # Build-step context (optional): compute indicator columns to show “how often will this trigger?”
    df_feat = None
    data_path_str = st.session_state.get("new.data_path")
    if data_path_str:
        try:
            p = Path(str(data_path_str))
            if p.exists():
                df_feat = _add_features_cached(str(p), p.stat().st_mtime)
        except Exception:
            df_feat = None

    build_atr_med = None

    colA, colB = st.columns(2)
    with colA:
        deposit_freq = st.selectbox(
            "Deposit frequency",
            options=["none", "daily", "weekly", "monthly"],
            index=2,
            key="new.deposit_freq",
        )
        deposit_amount = st.number_input(
            "Deposit amount (USD)",
            min_value=0.0,
            value=50.0,
            step=10.0,
            key="new.deposit_amount",
        )
        if str(deposit_freq).lower() == "none":
            deposit_amount = 0.0

        buy_freq = st.selectbox(
            "Buy frequency",
            options=["daily", "weekly", "monthly"],
            index=1,
            key="new.buy_freq",
        )
        buy_amount = st.number_input(
            "Buy amount (USD)",
            min_value=0.0,
            value=50.0,
            step=10.0,
            key="new.buy_amount",
        )

    # -------------------------
    # Entry logic (simple vs builder)
    # -------------------------
    with colB:
        st.caption("Entry gating controls whether scheduled buys are allowed to fire. This is about mechanics, not recommendations.")

        entry_mode = st.radio(
            "Entry logic mode",
            options=["Simple (one filter)", "Logic builder (regime + triggers)"],
            index=0,
            horizontal=True,
            key="new.entry_mode",
        )

        # Defaults for legacy knobs (used in simple mode and as defaults in the builder)
        buy_filter = "none"
        ema_len = 200
        rsi_thr = 40.0
        macd_hist_thr = 0.0
        bb_z_thr = -1.0
        adx_thr = 20.0
        donch_pos_thr = 0.20

        entry_logic: Dict[str, Any] = {"regime": [], "clauses": []}

        def _cond_ui(prefix: str) -> Optional[Dict[str, Any]]:
            """
            Builder UI for one condition. Returns condition dict or None if disabled.
            We keep condition types tight so we only reference known feature columns.
            """
            cond_type = st.selectbox(
                "Condition",
                options=[
                    "— (disabled)",
                    "Price vs EMA",
                    "RSI(14)",
                    "Bollinger z-score(20)",
                    "MACD histogram(12,26,9)",
                    "ADX(14)",
                    "Donchian position(20)",
                ],
                index=0,
                key=f"{prefix}.type",
            )
            if cond_type.startswith("—"):
                return None

            # Operators (restricted per type to reduce foot-guns)
            if cond_type == "Price vs EMA":
                op = st.selectbox("Operator", options=["<=", ">="], index=0, key=f"{prefix}.op")
                ln = int(st.selectbox("EMA length", options=[10, 20, 50, 100, 200], index=4, key=f"{prefix}.ema_len"))
                return {"indicator": "close", "operator": op, "ref_indicator": f"ema_{ln}", "threshold": 0.0}

            if cond_type == "RSI(14)":
                op = st.selectbox("Operator", options=["<=", ">="], index=0, key=f"{prefix}.op")
                thr = float(st.slider("Threshold", 5.0, 95.0, 40.0, 1.0, key=f"{prefix}.thr"))
                return {"indicator": "rsi_14", "operator": op, "threshold": thr}

            if cond_type == "Bollinger z-score(20)":
                op = st.selectbox("Operator", options=["<=", ">="], index=0, key=f"{prefix}.op")
                thr = float(st.slider("Threshold", -3.0, 3.0, -1.0, 0.1, key=f"{prefix}.thr"))
                return {"indicator": "bb_z_20", "operator": op, "threshold": thr}

            if cond_type == "MACD histogram(12,26,9)":
                op = st.selectbox("Operator", options=[">=", "<="], index=0, key=f"{prefix}.op")
                thr = float(st.number_input("Threshold", value=0.0, step=0.1, key=f"{prefix}.thr"))
                return {"indicator": "macd_hist_12_26_9", "operator": op, "threshold": thr}

            if cond_type == "ADX(14)":
                op = st.selectbox("Operator", options=[">=", "<="], index=0, key=f"{prefix}.op")
                thr = float(st.slider("Threshold", 1.0, 80.0, 20.0, 1.0, key=f"{prefix}.thr"))
                return {"indicator": "adx_14", "operator": op, "threshold": thr}

            if cond_type == "Donchian position(20)":
                op = st.selectbox("Operator", options=["<=", ">="], index=0, key=f"{prefix}.op")
                thr = float(st.slider("Threshold", 0.0, 1.0, 0.20, 0.05, key=f"{prefix}.thr"))
                return {"indicator": "donch_pos_20", "operator": op, "threshold": thr}

            return None

        if entry_mode.startswith("Simple"):
            FILTER_LABELS = {
                "none": "🟦 Always buy (no filter)",
                "below_ema": "📉 Buy dips below EMA",
                "rsi_below": "😮‍💨 Oversold (RSI low)",
                "bb_z_below": "🎯 Oversold (Bollinger stretch)",
                "macd_bull": "🚀 Momentum (MACD bullish)",
                "adx_above": "🧭 Trend strength (ADX)",
                "donch_pos_below": "🏷️ Range bottom (Donchian)",
            }
            FILTER_DESC = {
                "none": "Buys fire on schedule (subject to max allocation).",
                "below_ema": "Only buy when price is below a moving average (dip gate).",
                "rsi_below": "Only buy when RSI is below a threshold (oversold gate).",
                "bb_z_below": "Only buy when price is stretched below its Bollinger midline (z‑score).",
                "macd_bull": "Only buy when momentum is bullish (MACD histogram ≥ threshold).",
                "adx_above": "Only buy when trend strength is above a threshold (ADX).",
                "donch_pos_below": "Only buy near the bottom of the recent Donchian range.",
            }

            buy_filter = st.selectbox(
                "Entry filter (TradingView‑style)",
                options=[
                    "none",
                    "below_ema",
                    "rsi_below",
                    "bb_z_below",
                    "macd_bull",
                    "adx_above",
                    "donch_pos_below",
                ],
                index=0,
                key="new.buy_filter",
                format_func=lambda v: FILTER_LABELS.get(v, v),
            )
            st.caption(FILTER_DESC.get(buy_filter, ""))

            if buy_filter == "below_ema":
                ema_len = int(st.selectbox("EMA length", options=[10, 20, 50, 100, 200], index=4, key="new.ema_len"))
            elif buy_filter == "rsi_below":
                rsi_thr = float(st.slider("RSI threshold (buy when RSI ≤ threshold)", 5.0, 80.0, 40.0, 1.0, key="new.rsi_thr"))
            elif buy_filter == "bb_z_below":
                bb_z_thr = float(st.slider("Bollinger z-score threshold (buy when z ≤ threshold)", -3.0, 0.0, -1.0, 0.1, key="new.bb_z_thr"))
            elif buy_filter == "macd_bull":
                macd_hist_thr = float(st.number_input("MACD histogram threshold (hist ≥ threshold)", value=0.0, step=0.1, key="new.macd_hist_thr"))
            elif buy_filter == "adx_above":
                adx_thr = float(st.slider("ADX threshold (buy when ADX ≥ threshold)", 5.0, 60.0, 20.0, 1.0, key="new.adx_thr"))
            elif buy_filter == "donch_pos_below":
                donch_pos_thr = float(st.slider("Donchian position threshold (pos ≤ threshold)", 0.0, 1.0, 0.20, 0.05, key="new.donch_pos_thr"))

            entry_logic = _simple_filter_to_entry_logic(
                buy_filter,
                ema_len=ema_len,
                rsi_thr=rsi_thr,
                macd_hist_thr=macd_hist_thr,
                bb_z_thr=bb_z_thr,
                adx_thr=adx_thr,
                donch_pos_thr=donch_pos_thr,
            )

            # Tiny “reality check” in simple mode
            if df_feat is not None and not df_feat.empty:
                try:
                    pct = _filter_true_pct(
                        df_feat,
                        buy_filter,
                        ema_len=ema_len,
                        rsi_thr=rsi_thr,
                        macd_hist_thr=macd_hist_thr,
                        bb_z_thr=bb_z_thr,
                        adx_thr=adx_thr,
                        donch_pos_thr=donch_pos_thr,
                    )
                    atr_med = None
                    if "atr_pct" in df_feat.columns:
                        atr_med = float(np.nanmedian(pd.to_numeric(df_feat["atr_pct"], errors="coerce")))
                    build_atr_med = atr_med
                    msg = None
                    if pct is not None:
                        msg = f"On this dataset: filter true ~{pct:.0f}% of days"
                    if atr_med is not None and math.isfinite(atr_med):
                        msg = (msg + f" · median daily ATR ≈ {atr_med:.1f}%") if msg else f"Median daily ATR ≈ {atr_med:.1f}%"
                    if msg:
                        st.caption(msg + ".")
                    if pct is not None and pct < 5:
                        st.info("FYI: this filter triggers on <5% of days here. That typically reduces trade count and increases outcome variability.")
                except Exception:
                    pass

        else:
            st.caption("Builder: define a small set of gates (regime) and a few entry trigger clauses (any-of). Caps keep this auditable.")
            st.markdown("**Regime (0–2 gates, AND):** entries are allowed only when all regime gates are true.")
            reg_conds: List[Dict[str, Any]] = []
            with st.expander("Regime gates", expanded=True):
                c1 = _cond_ui("new.regime1")
                if c1:
                    reg_conds.append(c1)
                c2 = _cond_ui("new.regime2")
                if c2:
                    reg_conds.append(c2)

            st.markdown("**Entry triggers (1–3 clauses, OR-of-AND):** if *any* clause is true, an entry is allowed (subject to regime + allocation).")
            clauses: List[List[Dict[str, Any]]] = []
            with st.expander("Trigger clauses", expanded=True):
                for i in range(1, 4):
                    with st.container():
                        st.markdown(f"**Clause {i}** (1–3 conditions, AND)")
                        cl: List[Dict[str, Any]] = []
                        c1 = _cond_ui(f"new.cl{i}.c1")
                        if c1:
                            cl.append(c1)
                        c2 = _cond_ui(f"new.cl{i}.c2")
                        if c2:
                            cl.append(c2)
                        c3 = _cond_ui(f"new.cl{i}.c3")
                        if c3:
                            cl.append(c3)
                        if cl:
                            clauses.append(cl)
                        st.divider()

            if not clauses:
                st.warning("No trigger clause is enabled yet. Add at least one condition to at least one clause.")
            entry_logic = {"regime": reg_conds, "clauses": clauses}

            # Reality-check: how often does this fire?
            if df_feat is not None and not df_feat.empty and clauses:
                try:
                    pcts = _entry_logic_true_pcts(df_feat, entry_logic)
                    atr_med = None
                    if "atr_pct" in df_feat.columns:
                        atr_med = float(np.nanmedian(pd.to_numeric(df_feat["atr_pct"], errors="coerce")))
                    build_atr_med = atr_med
                    if pcts is not None:
                        r, e, c = pcts
                        msg = f"On this dataset: regime true ~{r:.0f}% · triggers true ~{e:.0f}% · combined ~{c:.0f}%"
                        if atr_med is not None and math.isfinite(atr_med):
                            msg += f" · median daily ATR ≈ {atr_med:.1f}%"
                        st.caption(msg + ".")
                        if c < 2:
                            st.info("FYI: combined entry gating triggers on <2% of days here. That often leads to very low trade counts and high variability.")
                except Exception:
                    pass

        max_alloc_pct = float(
            st.slider(
                "Max allocation (fraction of equity)",
                min_value=0.05,
                max_value=1.00,
                value=1.00,
                step=0.05,
                key="new.max_alloc_pct",
            )
        )

        # Defaults (controls disabled by default)
        sl_pct = 0.0
        tp_pct = 0.0
        tp_sell_fraction = 1.0
        reserve_frac = 0.0

        # New exits
        max_hold_bars = 0
        trail_pct = 0.0

        with st.expander("Risk controls (optional)", expanded=False):
            st.caption("Values are % moves from your average entry (10 = 10%). These are experiment knobs, not recommendations.")
            if build_atr_med is not None and math.isfinite(build_atr_med):
                st.caption(f"For scale only: median daily ATR ≈ {build_atr_med:.1f}% on this dataset.")

            sl_ui = float(st.slider("Stop loss (%) (0 disables)", 0.0, 95.0, 0.0, 0.25, key="new.sl_pct_ui"))
            tp_ui = float(st.slider("Take profit (%) (0 disables)", 0.0, 500.0, 0.0, 0.5, key="new.tp_pct_ui"))

            # New: time stop + trailing stop
            max_hold_bars = int(st.number_input("Time stop: max holding period (bars) (0 disables)", min_value=0, value=0, step=5, key="new.max_hold_bars"))
            trail_ui = float(st.slider("Trailing stop (%) from peak (0 disables)", 0.0, 95.0, 0.0, 0.25, key="new.trail_pct_ui"))

            if build_atr_med is not None and math.isfinite(build_atr_med) and build_atr_med > 0:
                if sl_ui > 0:
                    st.caption(f"Stop loss scale: {sl_ui:.2f}% ≈ {sl_ui / build_atr_med:.1f}× median ATR.")
                if tp_ui > 0:
                    st.caption(f"Take profit scale: {tp_ui:.2f}% ≈ {tp_ui / build_atr_med:.1f}× median ATR.")
                if trail_ui > 0:
                    st.caption(f"Trailing scale: {trail_ui:.2f}% ≈ {trail_ui / build_atr_med:.1f}× median ATR.")

            # Store as fractions in params (0.10 == 10%)
            sl_pct = sl_ui / 100.0
            tp_pct = tp_ui / 100.0
            trail_pct = trail_ui / 100.0

            if tp_ui > 0:
                tp_sell_fraction = float(st.slider("On take profit: sell fraction of position", 0.0, 1.0, 1.0, 0.05, key="new.tp_sell_frac"))
                reserve_frac = float(st.slider("Reserve fraction of TP proceeds (keep as cash)", 0.0, 1.0, 0.0, 0.05, key="new.reserve_frac"))
            else:
                tp_sell_fraction = 1.0
                reserve_frac = 0.0

    params = {
        "deposit_freq": deposit_freq,
        "deposit_amount_usd": float(deposit_amount),
        "buy_freq": buy_freq,
        "buy_amount_usd": float(buy_amount),
        "buy_filter": buy_filter,  # legacy (still used by grid/back-compat)
        "ema_len": int(ema_len),
        "rsi_thr": float(rsi_thr),
        "macd_hist_thr": float(macd_hist_thr),
        "bb_z_thr": float(bb_z_thr),
        "adx_thr": float(adx_thr),
        "donch_pos_thr": float(donch_pos_thr),
        "entry_logic": entry_logic,  # new
        "max_alloc_pct": float(max_alloc_pct),
        "sl_pct": float(sl_pct),
        "tp_pct": float(tp_pct),
        "tp_sell_fraction": float(tp_sell_fraction),
        "reserve_frac": float(reserve_frac),
        "max_hold_bars": int(max_hold_bars),
        "trail_pct": float(trail_pct),
    }

    # Human-readable strategy card (dopamine-friendly)
    parts: List[str] = []
    parts.append(f"Deposit {deposit_freq} ${int(round(params['deposit_amount_usd']))} and buy {params['buy_freq']} ${int(round(params['buy_amount_usd']))}.")
    if entry_mode.startswith("Simple"):
        if buy_filter == "below_ema":
            parts.append(f"Only buy if close ≤ EMA({params['ema_len']}).")
        elif buy_filter == "rsi_below":
            parts.append(f"Only buy if RSI(14) ≤ {params['rsi_thr']:.0f}.")
        elif buy_filter == "bb_z_below":
            parts.append(f"Only buy if BB z-score(20) ≤ {params['bb_z_thr']:.1f}.")
        elif buy_filter == "macd_bull":
            parts.append(f"Only buy if MACD histogram ≥ {params['macd_hist_thr']:.2f}.")
        elif buy_filter == "adx_above":
            parts.append(f"Only buy if ADX(14) ≥ {params['adx_thr']:.0f}.")
        elif buy_filter == "donch_pos_below":
            parts.append(f"Only buy near range bottom (Donchian pos ≤ {params['donch_pos_thr']:.2f}).")
        else:
            parts.append("Entry gate: (none).")
    else:
        parts.append(_human_entry_logic(entry_logic) or "Entry logic: (none).")

    parts.append(f"Never allocate more than {_fmt_pct(params['max_alloc_pct'], digits=0)} of equity.")

    # Risk controls
    if params.get("sl_pct", 0.0) > 0:
        parts.append("Stop loss: {:.2f}%.".format(params["sl_pct"] * 100))
    if params.get("tp_pct", 0.0) > 0:
        parts.append("Take profit: {:.2f}% (sell {}% per hit).".format(params["tp_pct"] * 100, int(round(params["tp_sell_fraction"] * 100))))
    if params.get("max_hold_bars", 0) > 0:
        parts.append(f"Time stop: exit after {params['max_hold_bars']} bars in-position.")
    if params.get("trail_pct", 0.0) > 0:
        parts.append("Trailing stop: {:.2f}% from peak (ratchets up).".format(params["trail_pct"] * 100))

    st.info(" ".join([p for p in parts if p.strip()]))

    return params

def _write_baseline_json(tmp_dir: Path, *, strategy_name: str, side: str, params: Dict[str, Any]) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    p = tmp_dir / f"baseline_{_now_slug()}.json"
    cfg = {
        "strategy_name": strategy_name,
        "side": side,
        "params": params,
    }
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return p


def _read_json(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(',', ':')))
            f.write('\n')


# =============================================================================
# Trust layer (Sprint 4): manifest + comparability + strategy pack export
# =============================================================================

MANIFEST_SCHEMA_VERSION = 2


def _utc_iso(ts: float) -> str:
    try:
        return datetime.utcfromtimestamp(float(ts)).replace(microsecond=0).isoformat() + "Z"
    except Exception:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _try_git_head(repo_root: Path) -> Optional[str]:
    """Best-effort git commit hash for receipts."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL)
        s = out.decode("utf-8", errors="ignore").strip()
        return s if re.fullmatch(r"[0-9a-fA-F]{7,40}", s or "") else None
    except Exception:
        return None


def _fingerprint_file(path: Path, *, full_max_bytes: int = 50_000_000, sample_bytes: int = 1_000_000) -> Dict[str, Any]:
    """
    Compute a stable-ish dataset fingerprint.
    - Full sha256 for small files.
    - Head+tail sha256 for large files.
    """
    stt = path.stat()
    size = int(stt.st_size)
    mtime = float(stt.st_mtime)

    mode = "full" if size <= int(full_max_bytes) else "sample"
    h = hashlib.sha256()

    with open(path, "rb") as f:
        if mode == "full":
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        else:
            head = f.read(int(sample_bytes))
            h.update(head)
            if size > int(sample_bytes):
                try:
                    f.seek(max(0, size - int(sample_bytes)))
                    tail = f.read(int(sample_bytes))
                    h.update(tail)
                except Exception:
                    pass

    fp: Dict[str, Any] = {
        "algo": "sha256",
        "mode": mode,
        "digest": h.hexdigest(),
        "size_bytes": size,
        "mtime_utc": _utc_iso(mtime),
    }
    if mode == "sample":
        fp["sample_bytes"] = int(sample_bytes)
    return fp





# -----------------------------------------------------------------------------
# Trust layer helpers (Sprint 5)
# -----------------------------------------------------------------------------
def _safe_relpath(path: Optional[Path], base: Path) -> Optional[str]:
    try:
        if path is None:
            return None
        return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return None


def _dataset_quick_meta(path: Path, *, max_scan_bytes: int = 200_000_000, tail_lines: int = 2000) -> Dict[str, Any]:
    """Best-effort dataset metadata for comparability (fast-ish, guarded for huge files)."""
    meta: Dict[str, Any] = {}
    try:
        stt = path.stat()
        if int(stt.st_size) > int(max_scan_bytes):
            meta["note"] = "Skipped deep scan (file too large)."
            return meta
    except Exception:
        return meta

    # Columns + schema hash + row count
    try:
        import csv

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                meta["columns"] = header
                meta["schema_hash"] = hashlib.sha256(",".join(header).encode("utf-8")).hexdigest()
                cnt = 0
                for _ in reader:
                    cnt += 1
                meta["row_count"] = int(cnt)
    except Exception:
        pass

    # Time range hint (head/tail sampling)
    try:
        head: List[str] = []
        tail: deque[str] = deque(maxlen=int(tail_lines))
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i < 2000:
                    head.append(line)
                tail.append(line)

        cols = meta.get("columns") or []
        cand = None
        for c in ["ts", "timestamp", "dt", "date", "datetime"]:
            if c in cols:
                cand = c
                break

        if cand:
            def _parse(lines: List[str]) -> Tuple[Optional[float], Optional[float]]:
                if not lines:
                    return (None, None)
                try:
                    df = pd.read_csv(io.StringIO("".join(lines)))
                    if cand not in df.columns or df.empty:
                        return (None, None)
                    ser = df[cand]
                    if cand in ("ts", "timestamp"):
                        v = pd.to_numeric(ser, errors="coerce").dropna()
                        if v.empty:
                            return (None, None)
                        return (float(v.min()), float(v.max()))
                    v = pd.to_datetime(ser, errors="coerce", utc=True).dropna()
                    if v.empty:
                        return (None, None)
                    return (float(v.min().timestamp()), float(v.max().timestamp()))
                except Exception:
                    return (None, None)

            h0, h1 = _parse(head)
            t0, t1 = _parse(list(tail))
            xs = [x for x in [h0, h1, t0, t1] if x is not None]
            if xs:
                meta["time_range_hint_epoch"] = {"min": min(xs), "max": max(xs), "column": cand}
    except Exception:
        pass

    return meta


def _tests_signature(manifest: Dict[str, Any]) -> Dict[str, Any]:
    tests = (manifest or {}).get("tests") or {}
    rs_runs = tests.get("rolling_starts") or []
    wf_runs = tests.get("walkforward") or []

    def _pick_rs(m: Dict[str, Any]) -> Dict[str, Any]:
        return {"start_step": m.get("start_step"), "min_bars": m.get("min_bars"), "top_n": m.get("top_n"), "windows_per_cfg": m.get("windows_per_cfg")}

    def _pick_wf(m: Dict[str, Any]) -> Dict[str, Any]:
        return {"window_days": m.get("window_days"), "step_days": m.get("step_days"), "min_bars": m.get("min_bars"), "top_n": m.get("top_n"), "windows": m.get("windows")}

    rs_sigs = []
    for r in rs_runs:
        if isinstance(r, dict):
            rs_sigs.append(_pick_rs((r.get("meta") or {})))
    wf_sigs = []
    for r in wf_runs:
        if isinstance(r, dict):
            wf_sigs.append(_pick_wf((r.get("meta") or {})))

    def _uniq(xs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for x in xs:
            key = json.dumps(x, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                out.append(x)
        return out

    return {"rolling_starts": _uniq(rs_sigs), "walkforward": _uniq(wf_sigs)}


def _compare_manifests(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    warns: List[str] = []

    da = ((a or {}).get("dataset") or {}).get("fingerprint") or {}
    db = ((b or {}).get("dataset") or {}).get("fingerprint") or {}
    if str(da.get("digest") or "") and str(db.get("digest") or "") and str(da.get("digest")) != str(db.get("digest")):
        warns.append("Dataset fingerprints differ between runs (results are not directly comparable).")

    ga = str((a or {}).get("engine_git_head") or "")
    gb = str((b or {}).get("engine_git_head") or "")
    if ga and gb and ga != gb:
        warns.append("Engine git differs between runs (behavior may differ).")

    sa = _tests_signature(a)
    sb = _tests_signature(b)
    if json.dumps(sa.get("rolling_starts"), sort_keys=True) != json.dumps(sb.get("rolling_starts"), sort_keys=True):
        warns.append("Rolling Starts parameters differ between runs.")
    if json.dumps(sa.get("walkforward"), sort_keys=True) != json.dumps(sb.get("walkforward"), sort_keys=True):
        warns.append("Walkforward parameters differ between runs.")

    return warns


def _update_runs_index(runs_root: Path) -> None:
    """Maintain runs/_index.json for quick browsing + cross-run compare."""
    items: List[Dict[str, Any]] = []
    try:
        for d in sorted([p for p in runs_root.glob("batch_*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
            mp = d / "manifest.json"
            if not mp.exists():
                continue
            m = _read_json(mp)
            ds = (m.get("dataset") or {}) if isinstance(m, dict) else {}
            fp = (ds.get("fingerprint") or {}) if isinstance(ds, dict) else {}
            dig = str(fp.get("digest") or "")
            items.append(
                {
                    "run_id": d.name,
                    "created_at": m.get("created_at") if isinstance(m, dict) else _utc_iso(d.stat().st_mtime),
                    "dataset_digest": dig,
                    "dataset_digest_short": (dig[:10] + "…" + dig[-6:]) if dig else "",
                    "engine_git_head": (m.get("engine_git_head") if isinstance(m, dict) else None) or "",
                    "tests": _tests_signature(m if isinstance(m, dict) else {}),
                }
            )
    except Exception:
        pass

    try:
        out = {"schema_version": 1, "updated_at": _utc_iso(time.time()), "items": items}
        _write_json(runs_root / "_index.json", out)
    except Exception:
        pass
def _scan_test_runs(run_dir: Path) -> Dict[str, Any]:
    """Scan RS/WF output folders and collect meta for receipts + warnings."""
    rs_root = run_dir / "rolling_starts"
    wf_root = run_dir / "walkforward"

    def _dirs(root: Path, pat: str) -> List[Path]:
        if not root.exists():
            return []
        xs = [p for p in root.glob(pat) if p.is_dir()]
        return sorted(xs, key=lambda p: p.stat().st_mtime)

    rs_runs: List[Dict[str, Any]] = []
    for d in _dirs(rs_root, "rs_*"):
        rs_runs.append(
            {
                "dir": str(d),
                "meta": _read_json(d / "rs_meta.json") if (d / "rs_meta.json").exists() else {},
                "summary": str(d / "rolling_starts_summary.csv") if (d / "rolling_starts_summary.csv").exists() else None,
                "detail": str(d / "rolling_starts_detail.csv") if (d / "rolling_starts_detail.csv").exists() else None,
                "mtime_utc": _utc_iso(d.stat().st_mtime),
            }
        )

    wf_runs: List[Dict[str, Any]] = []
    for d in _dirs(wf_root, "wf_*"):
        wf_runs.append(
            {
                "dir": str(d),
                "meta": _read_json(d / "wf_meta.json") if (d / "wf_meta.json").exists() else {},
                "summary": str(d / "wf_summary.csv") if (d / "wf_summary.csv").exists() else None,
                "results": str(d / "wf_results.csv") if (d / "wf_results.csv").exists() else None,
                "mtime_utc": _utc_iso(d.stat().st_mtime),
            }
        )

    return {"rolling_starts": rs_runs, "walkforward": wf_runs}


def _build_manifest(
    run_dir: Path,
    *,
    compute_fingerprint: bool = True,
    existing_fp: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta = _read_json(run_dir / "batch_meta.json") if (run_dir / "batch_meta.json").exists() else {}
    data_path = Path(str(meta.get("data") or meta.get("data_path") or "")).expanduser()
    dataset: Dict[str, Any] = {
        "path_abs": str(data_path) if str(data_path) else None,
        "path_rel_to_repo": _safe_relpath(data_path, REPO_ROOT) if str(data_path) else None,
        "basename": data_path.name if str(data_path) else None,
    }

    if str(data_path) and data_path.exists():
        # Avoid re-hashing on every UI rerun if an existing fingerprint is still valid.
        if (not compute_fingerprint) and isinstance(existing_fp, dict) and existing_fp:
            dataset["fingerprint"] = existing_fp
        else:
            dataset["fingerprint"] = _fingerprint_file(data_path)

        # Extra metadata for comparability (best-effort; guarded for huge files)
        try:
            dataset["meta"] = _dataset_quick_meta(data_path)
        except Exception:
            dataset["meta"] = {}

    created_guess = None
    # Try parse run folder name: batch_YYYYMMDD_HHMMSS_...
    m = re.match(r"batch_(\d{8})_(\d{6})_", run_dir.name)
    if m:
        try:
            dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            created_guess = dt.replace(tzinfo=None).isoformat() + "Z"
        except Exception:
            created_guess = None

    app_fp = None
    try:
        app_fp = _fingerprint_file(Path(__file__).resolve(), full_max_bytes=5_000_000)
    except Exception:
        app_fp = None

    run_dir_abs = run_dir.resolve()
    manifest: Dict[str, Any] = {
        "schema_version": int(MANIFEST_SCHEMA_VERSION),
        "run_id": str(run_dir.name),
        "created_at": created_guess or _utc_iso(run_dir.stat().st_mtime),
        "repo_root": str(REPO_ROOT),
        "run_dir": {"abs": str(run_dir_abs), "rel_to_repo": _safe_relpath(run_dir_abs, REPO_ROOT)},
        "engine_git_head": _try_git_head(REPO_ROOT),
        "app_file_fingerprint": app_fp,
        "dataset": dataset,
        "batch_meta": meta,
        "tests": _scan_test_runs(run_dir),
    }
    return manifest
def _ensure_manifest(run_dir: Path) -> Dict[str, Any]:
    """Create/refresh manifest.json. Safe for old runs; avoids re-hashing unchanged datasets."""
    path = run_dir / "manifest.json"
    try:
        existing = _read_json(path) if path.exists() else {}
    except Exception:
        existing = {}

    # Decide whether we need to recompute the dataset hash (can be expensive for big CSVs).
    compute_fp = True
    existing_fp = None
    try:
        ds = (existing or {}).get("dataset") or {}
        existing_fp = (ds.get("fingerprint") or {}) if isinstance(ds, dict) else None
        p = ds.get("path_abs") or ds.get("path")
        if p and isinstance(existing_fp, dict) and existing_fp:
            data_path = Path(str(p)).expanduser()
            if data_path.exists():
                stt = data_path.stat()
                cur_size = int(stt.st_size)
                cur_mtime = _utc_iso(float(stt.st_mtime))
                rec_size = int(existing_fp.get("size_bytes", -1))
                rec_mtime = str(existing_fp.get("mtime_utc") or "")
                if (rec_size != -1 and cur_size == rec_size) and (rec_mtime and cur_mtime == rec_mtime):
                    compute_fp = False
    except Exception:
        compute_fp = True
        existing_fp = None

    new = _build_manifest(run_dir, compute_fingerprint=compute_fp, existing_fp=existing_fp)

    # If an existing manifest exists, keep any unknown top-level keys.
    merged = dict(existing or {})
    merged.update(new)

    try:
        _write_json(path, merged)
    except Exception:
        # If writing fails (permissions), still return what we built.
        return merged

    # Update runs index (best-effort)
    try:
        _update_runs_index(REPO_ROOT / "runs")
    except Exception:
        pass

    return merged
def _comparability_warnings(manifest: Dict[str, Any]) -> List[str]:
    warns: List[str] = []

    ds = (manifest or {}).get("dataset") or {}
    p = ds.get("path_abs") or ds.get("path")
    fp = (ds.get("fingerprint") or {}) if isinstance(ds, dict) else {}

    if not p:
        warns.append("No dataset path recorded in manifest. Comparability is weaker.")
        return warns

    data_path = Path(str(p)).expanduser()
    if not data_path.exists():
        warns.append("Dataset file no longer exists at the recorded path. Comparability is weaker.")
        return warns

    # Quick check: size + mtime
    try:
        stt = data_path.stat()
        cur_size = int(stt.st_size)
        cur_mtime = _utc_iso(float(stt.st_mtime))
        rec_size = int(fp.get("size_bytes", -1)) if isinstance(fp, dict) else -1
        rec_mtime = str(fp.get("mtime_utc") or "")
        drift = False
        if rec_size != -1 and cur_size != rec_size:
            warns.append("Dataset size differs from the recorded fingerprint (file may have changed).")
            drift = True
        if rec_mtime and cur_mtime != rec_mtime:
            warns.append("Dataset modification time differs from the recorded fingerprint (file may have changed).")
            drift = True
        # Only compute a hash comparison if cheap checks suggest drift.
        if drift:
            cur_fp = _fingerprint_file(data_path)
            rec_digest = str(fp.get("digest") or "")
            if rec_digest and str(cur_fp.get("digest")) != rec_digest:
                warns.append("Dataset fingerprint digest does not match (you are not running on the same data).")
    except Exception:
        warns.append("Could not validate dataset fingerprint. Comparability is weaker.")

    # Multiple RS/WF parameter sets
    tests = (manifest or {}).get("tests") or {}
    rs_runs = tests.get("rolling_starts") or []
    wf_runs = tests.get("walkforward") or []

    def _rs_sig(m: Dict[str, Any]) -> str:
        return f"step={m.get('start_step')}|min={m.get('min_bars')}|top_n={m.get('top_n')}|wins={m.get('windows_per_cfg')}"

    def _wf_sig(m: Dict[str, Any]) -> str:
        return f"win={m.get('window_days')}|step={m.get('step_days')}|min={m.get('min_bars')}|top_n={m.get('top_n')}|wins={m.get('windows')}"

    try:
        rs_sigs = {_rs_sig((r.get("meta") or {})) for r in rs_runs if isinstance(r, dict)}
        rs_sigs = {s for s in rs_sigs if "None" not in s}
        if len(rs_sigs) > 1:
            warns.append("Multiple Rolling Starts evidence sets exist with different parameters. Verdicts depend on which evidence set is used.")
    except Exception:
        pass

    try:
        wf_sigs = {_wf_sig((r.get("meta") or {})) for r in wf_runs if isinstance(r, dict)}
        wf_sigs = {s for s in wf_sigs if "None" not in s}
        if len(wf_sigs) > 1:
            warns.append("Multiple Walkforward evidence sets exist with different parameters. Verdicts depend on which evidence set is used.")
    except Exception:
        pass

    return warns


def _zip_add_bytes(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    zf.writestr(arcname, data)


def _zip_add_file(zf: zipfile.ZipFile, file_path: Path, arcname: str) -> None:
    try:
        zf.write(str(file_path), arcname=arcname)
    except Exception:
        # fallback: read bytes
        try:
            _zip_add_bytes(zf, arcname, file_path.read_bytes())
        except Exception:
            pass


def _build_strategy_pack_zip(
    *,
    run_dir: Path,
    run_name: str,
    config_id: str,
    manifest: Dict[str, Any],
    candidate_row: Dict[str, Any],
    cfg_norm: Dict[str, Any],
    rs_dir: Optional[Path],
    wf_dir: Optional[Path],
    top_art_dir: Optional[Path],
    include_replay: bool = True,
    include_dataset: bool = False,
) -> bytes:
    """Strategy Pack v2: structured, portable, verifiable."""
    buf = io.BytesIO()
    index: Dict[str, Any] = {"pack_version": 2, "created_at": _utc_iso(time.time()), "files": {}}

    def _sha256_bytes(b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()

    def add_bytes(arc: str, b: bytes) -> None:
        _zip_add_bytes(zf, arc, b)
        index["files"][arc] = {"algo": "sha256", "digest": _sha256_bytes(b), "size_bytes": int(len(b))}

    def add_file(fp: Path, arc: str, *, hash_limit_bytes: int = 50_000_000) -> None:
        try:
            _zip_add_file(zf, fp, arc)
        except Exception:
            return
        try:
            size = int(fp.stat().st_size)
            digest = ""
            if size <= int(hash_limit_bytes):
                h = hashlib.sha256()
                with fp.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                digest = h.hexdigest()
            index["files"][arc] = {"algo": "sha256", "digest": digest, "size_bytes": size}
        except Exception:
            pass

    # Portable pack manifest (no absolute paths)
    ds = (manifest.get("dataset") or {}) if isinstance(manifest, dict) else {}
    ds_fp = (ds.get("fingerprint") or {}) if isinstance(ds, dict) else {}
    ds_meta = (ds.get("meta") or {}) if isinstance(ds, dict) else {}

    pack_manifest: Dict[str, Any] = {
        "schema_version": 2,
        "pack_version": 2,
        "source_run_id": str(run_name),
        "created_at": _utc_iso(time.time()),
        "engine_git_head": (manifest.get("engine_git_head") if isinstance(manifest, dict) else None) or "",
        "app_file_fingerprint": (manifest.get("app_file_fingerprint") if isinstance(manifest, dict) else None) or {},
        "dataset": {
            "basename": ds.get("basename") or (Path(str(ds.get("path_abs") or ds.get("path") or "")).name if (ds.get("path_abs") or ds.get("path")) else None),
            "fingerprint": ds_fp,
            "meta": ds_meta,
            "included_in_pack": bool(include_dataset),
        },
        "selected_config_id": str(config_id),
        "tests_signature": _tests_signature(manifest if isinstance(manifest, dict) else {}),
        "notes": "Portable strategy pack. Absolute paths removed; verify dataset via fingerprint.",
    }

    # README
    readme = f"""# Strategy Pack (v2)

This bundle contains receipts for a single strategy config.

- Source run: `{run_name}`
- Config: `{str(config_id)}`
- Engine git: `{pack_manifest.get('engine_git_head','')[:12]}`

## Verify
1. Use the in-app verifier to validate file hashes.
2. To verify your dataset, compare its fingerprint digest to `manifest.json -> dataset -> fingerprint -> digest`.

"""

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        add_bytes("README.md", readme.encode("utf-8"))
        add_bytes("manifest.json", json.dumps(pack_manifest, indent=2, ensure_ascii=False).encode("utf-8"))

        # Configs + evidence
        add_bytes("config/config_normalized.json", json.dumps(cfg_norm or {}, indent=2, ensure_ascii=False).encode("utf-8"))
        add_bytes("evidence/candidate_row.json", json.dumps(candidate_row or {}, indent=2, ensure_ascii=False).encode("utf-8"))

        # Resolved config line
        try:
            cfg_line = next((r for r in _load_jsonl(run_dir / "configs_resolved.jsonl") if str(r.get("config_id")) == str(config_id)), None)
            if cfg_line is not None:
                add_bytes("config/config_resolved.json", json.dumps(cfg_line, indent=2, ensure_ascii=False).encode("utf-8"))
        except Exception:
            pass

        # Batch meta
        if (run_dir / "batch_meta.json").exists():
            add_file(run_dir / "batch_meta.json", "meta/batch_meta.json")

        # RS evidence (filtered)
        if rs_dir is not None and rs_dir.exists():
            if (rs_dir / "rs_meta.json").exists():
                add_file(rs_dir / "rs_meta.json", "meta/rolling_starts/rs_meta.json")
            try:
                s = rs_dir / "rolling_starts_summary.csv"
                d = rs_dir / "rolling_starts_detail.csv"
                if s.exists():
                    sdf = pd.read_csv(s)
                    sdf["config_id"] = sdf["config_id"].astype(str).str.strip()
                    sdf1 = sdf[sdf["config_id"] == str(config_id)].copy()
                    add_bytes("evidence/rolling_starts/summary_row.csv", sdf1.to_csv(index=False).encode("utf-8"))
                if d.exists():
                    ddf = pd.read_csv(d)
                    ddf["config_id"] = ddf["config_id"].astype(str).str.strip()
                    ddf1 = ddf[ddf["config_id"] == str(config_id)].copy()
                    add_bytes("evidence/rolling_starts/detail_rows.csv", ddf1.to_csv(index=False).encode("utf-8"))
            except Exception:
                pass

        # WF evidence (filtered)
        if wf_dir is not None and wf_dir.exists():
            if (wf_dir / "wf_meta.json").exists():
                add_file(wf_dir / "wf_meta.json", "meta/walkforward/wf_meta.json")
            try:
                s = wf_dir / "wf_summary.csv"
                r = wf_dir / "wf_results.csv"
                if s.exists():
                    sdf = pd.read_csv(s)
                    sdf["config_id"] = sdf["config_id"].astype(str).str.strip()
                    sdf1 = sdf[sdf["config_id"] == str(config_id)].copy()
                    add_bytes("evidence/walkforward/summary_row.csv", sdf1.to_csv(index=False).encode("utf-8"))
                if r.exists():
                    rdf = pd.read_csv(r)
                    rdf["config_id"] = rdf["config_id"].astype(str).str.strip()
                    rdf1 = rdf[rdf["config_id"] == str(config_id)].copy()
                    add_bytes("evidence/walkforward/window_rows.csv", rdf1.to_csv(index=False).encode("utf-8"))
            except Exception:
                pass

        # Replay/top artifacts
        if include_replay and top_art_dir is not None and top_art_dir.exists():
            for fp in top_art_dir.rglob("*"):
                if fp.is_file():
                    rel = fp.relative_to(top_art_dir).as_posix()
                    add_file(fp, f"artifacts/{rel}", hash_limit_bytes=10_000_000)

        # Optional: include dataset
        if include_dataset:
            try:
                ds_path = ds.get("path_abs") or ds.get("path")
                if ds_path:
                    p = Path(str(ds_path)).expanduser()
                    if p.exists():
                        add_file(p, f"dataset/{p.name}", hash_limit_bytes=200_000_000)
            except Exception:
                pass

        # Pack index last
        add_bytes("meta/pack_index.json", json.dumps(index, indent=2, ensure_ascii=False).encode("utf-8"))

    return buf.getvalue()
def _baseline_row_from_base_json(base_path: Path) -> Dict[str, Any]:
    base = _read_json(base_path)
    if 'params' not in base or not isinstance(base['params'], dict):
        raise ValueError("Baseline JSON must contain a 'params' object")
    row = {
        'strategy_name': base.get('strategy_name', 'dca_swing'),
        'side': base.get('side', 'long'),
        'params': dict(base['params']),
    }
    # Tag so the UI can identify the baseline row in batch outputs
    row['params']['__baseline__'] = True
    return row

def _ensure_grid_has_baseline(grid_path: Path, base_path: Path, *, total_n: int) -> None:
    """Prepend baseline row to an existing grid JSONL (deduping any existing baseline row)."""
    baseline_row = _baseline_row_from_base_json(base_path)
    rows: List[Dict[str, Any]] = []
    if grid_path.exists():
        rows = [r for r in _load_jsonl(grid_path) if isinstance(r, dict)]
    # Drop any existing baseline row(s) to avoid duplicates
    cleaned: List[Dict[str, Any]] = []
    for r in rows:
        try:
            if isinstance(r.get('params'), dict) and r['params'].get('__baseline__'):
                continue
        except Exception:
            pass
        cleaned.append(r)
    out = [baseline_row, *cleaned]
    if int(total_n) > 0:
        out = out[: int(total_n)]
    _write_jsonl(grid_path, out)


# =============================================================================
# UI: left rail (runs + mode)
# =============================================================================

with st.sidebar:
    st.header("Runs")
    runs = _list_runs()
    run_names = [p.name for p in runs]
    run_dirs = {p.name: p for p in runs}
    run_is_complete = {name: _has_any_results(run_dirs[name]) for name in run_names}

    # Persist selection across reruns (prefer the latest complete run)
    if "selected_run" not in st.session_state:
        picked = ""
        for nm in run_names:
            if run_is_complete.get(nm):
                picked = nm
                break
        st.session_state["selected_run"] = picked or (run_names[0] if run_names else "")

    # Programmatic selection handoff (must happen BEFORE the widget is created)
    nxt = st.session_state.pop("ui.open_run_next", None)
    if nxt and nxt in run_names:
        st.session_state["selected_run"] = nxt

    open_existing = st.selectbox(
        "Open existing run",
        options=["(new run)"] + run_names,
        index=(1 + run_names.index(st.session_state["selected_run"]) if st.session_state["selected_run"] in run_names else 0),
        format_func=lambda nm: (nm if nm == "(new run)" else (nm + ("  (incomplete)" if not run_is_complete.get(nm, False) else ""))),
        key="ui.open_run",
    )

    if open_existing != "(new run)":
        st.session_state["selected_run"] = open_existing
    else:
        # Selecting "(new run)" should always drop you into the Build & Run flow.
        st.session_state["ui.section"] = "1) Build & Run"

    st.divider()
    st.header("App")

    st.session_state.setdefault("ui.debug", False)
    st.checkbox(
        "Debug (show commands & logs)",
        key="ui.debug",
        help="Off by default to keep the UI clean. Turn on to see raw subprocess logs and full commands.",
    )


    # Keep stage keys for internal routing ("Next →" buttons, etc.)
    STAGES = [
        ("A) Batch", "batch"),
        ("B) Rolling Starts", "rs"),
        ("C) Walkforward", "wf"),
        ("D) Grand Verdict", "grand"),
    ]
    stage_labels = [x[0] for x in STAGES]
    stage_keys = [x[1] for x in STAGES]

    st.session_state.setdefault("ui.stage", "batch")
    st.session_state.setdefault("ui.batch.scroll_to_inspect", False)

    # MVP navigation: two sections only
    SECTION_OPTS = ["1) Build & Run", "2) Results & Autopsy"]
    if "ui.section" not in st.session_state:
        st.session_state["ui.section"] = SECTION_OPTS[0] if open_existing == "(new run)" else SECTION_OPTS[1]

    if "ui.section_next" in st.session_state:
        st.session_state["ui.section"] = st.session_state.pop("ui.section_next")

    st.radio("Section", options=SECTION_OPTS, key="ui.section")
    st.caption("Build & Run = define strategy + run tests. Results & Autopsy = filter, compare, inspect.")

# =============================================================================
# New run wizard (when "(new run)" is selected)
# =============================================================================

if open_existing == "(new run)":
    # This section is only the "Build & Run" half of the MVP UI.
    if str(st.session_state.get("ui.section", "1) Build & Run")).startswith("2)"):
        st.info("Results require an existing run. Switch to **Build & Run** to create one.")
        st.stop()
    st.subheader("Create a new run")

    # Step state
    if "new.step" not in st.session_state:
        st.session_state["new.step"] = 0  # 0=data,1=plan,2=grid,3=batch

    NEW_STEPS = ["1) Data", "2) Baseline plan", "3) Variations", "4) Run batch"]
    step = int(st.session_state["new.step"])
    step = max(0, min(step, len(NEW_STEPS) - 1))

    st.progress((step + 1) / len(NEW_STEPS))
    st.write(f"**{NEW_STEPS[step]}**")

    # Shared: staging dir for temp files
    tmp_dir = REPO_ROOT / ".ui_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 0: Data
    # -------------------------------------------------------------------------
    if step == 0:
        st.write("Pick the dataset you want to test against (daily OHLCV for spot).")

        uploaded = st.file_uploader("Upload OHLCV CSV", type=["csv"], key="new.upload")

        sample_csv = DATA_DIR / "eth_daily_2023_to_now.csv"
        use_sample = st.checkbox(
            f"Use sample dataset ({sample_csv.name})",
            value=(uploaded is None and sample_csv.exists()),
            key="new.use_sample",
        )

        data_path: Optional[Path] = None
        if use_sample and sample_csv.exists():
            data_path = sample_csv
        elif uploaded is not None:
            out = tmp_dir / f"upload_{_now_slug()}_{_slug(uploaded.name)}"
            out.write_bytes(uploaded.getvalue())
            data_path = out

        if data_path is not None:
            st.session_state["new.data_path"] = str(data_path)

            df_preview = _load_csv(data_path)
            if df_preview is not None and not df_preview.empty:
                st.caption(f"Rows: {len(df_preview):,}  Columns: {list(df_preview.columns)}")
                # Light chart if possible
                if px is not None:
                    # Try to detect a date column
                    dt_col = None
                    for c in ["dt", "date", "timestamp", "ts"]:
                        if c in df_preview.columns:
                            dt_col = c
                            break
                    if dt_col:
                        try:
                            dfx = df_preview.copy()
                            dfx[dt_col] = pd.to_datetime(dfx[dt_col], errors="coerce", utc=True)
                            dfx = dfx.dropna(subset=[dt_col])
                            if "close" in dfx.columns:
                                fig = px.line(dfx.tail(2000), x=dt_col, y="close", title="Close (tail)")
                                _plotly(fig)
                        except Exception:
                            pass
            else:
                st.warning("Could not preview dataset (CSV parse failed).")

        colL, colR = st.columns(2)
        with colL:
            if st.button("Next →", type="primary", disabled=("new.data_path" not in st.session_state)):
                st.session_state["new.step"] = 1
                st.rerun()

    # -------------------------------------------------------------------------
    # Step 1: Plan
    # -------------------------------------------------------------------------
    elif step == 1:
        st.write("Define your plan. This becomes the baseline that variations are generated around.")

        strategy_template = st.selectbox(
            "Strategy template",
            options=[
                "DCA/Swing (beginner-friendly, long-only)",
            ],
            index=0,
            key="new.template",
        )

        baseline_params = build_dca_baseline_params()

        st.session_state["new.baseline_params"] = baseline_params
        st.session_state["new.template_path"] = "strategies.dca_swing:Strategy"
        st.session_state["new.grid_script"] = str(REPO_ROOT / "tools" / "make_dca_grid.py")
        st.session_state["new.market_mode"] = "spot"
        st.session_state["new.strategy_name"] = "dca_swing"

        colL, colR = st.columns(2)
        with colL:
            if st.button("← Back"):
                st.session_state["new.step"] = 0
                st.rerun()
        with colR:
            if st.button("Next →", type="primary"):
                st.session_state["new.step"] = 2
                st.rerun()

    # -------------------------------------------------------------------------
    # Step 2: Variations (grid)
    # -------------------------------------------------------------------------
    elif step == 2:
        st.write("Decide how you want to stress test the baseline plan.")

        mode = st.selectbox(
            "How to generate variants",
            options=[
                "Stress test my plan (recommended)",
                "Explore random variations (advanced)",
            ],
            index=0,
            key="new.grid_mode",
        )

        # Friendly presets → translates to make_dca_grid width + N
        preset = st.selectbox(
            "Stress level",
            options=["Low", "Medium", "High"],
            index=1,
            key="new.stress_level",
        )
        preset_map = {
            "Low": ("narrow", 300),
            "Medium": ("medium", 1000),
            "High": ("wide", 2000),
        }
        width, n_default = preset_map[preset]

        n = int(st.number_input("Number of variants", min_value=50, max_value=10000, value=int(n_default), step=50, key="new.n"))
        seed = int(st.number_input("Random seed", min_value=1, max_value=10_000_000, value=1, step=1, key="new.seed"))

                # Decide which vary-groups to expose based on the baseline entry mode.
        entry_mode_ui = str(st.session_state.get("new.entry_mode", ""))
        entry_group = "logic" if entry_mode_ui.startswith("Logic builder") else "filter"

        vary_groups = ["deposits", "buys", entry_group, "alloc", "risk"]
        vary = list(vary_groups)

        LABELS = {
            "deposits": "Deposits",
            "buys": "Buys",
            "filter": "Entry filter (simple)",
            "logic": "Entry logic (builder)",
            "alloc": "Allocation cap",
            "risk": "Risk & exits (SL/TP + time/trail)",
        }

        logic_frac = float(st.session_state.get("new.logic_frac", 0.35))

        if mode.startswith("Stress test"):
            st.write("What should be allowed to vary around the baseline?")
            cols = st.columns(len(vary_groups))
            picks: List[str] = []
            for i, g in enumerate(vary_groups):
                with cols[i]:
                    if st.checkbox(LABELS.get(g, g), value=True, key=f"new.vary.{g}"):
                        picks.append(g)
            vary = picks or ["deposits", "buys"]  # never allow empty

        else:
            # Random mode ignores vary/width/base; we do allow controlling the mix of logic-builder vs simple entry configs.
            st.caption("Random mode explores a broader space. This slider controls how often the generator uses the logic-builder entry mode.")
            logic_frac = float(st.slider("Logic-builder share", 0.0, 1.0, float(logic_frac), 0.05, key="new.logic_frac"))
            vary = list(vary_groups)

        # Preview
        st.divider()
        st.write("Preview (optional)")
        if st.button("Generate a preview grid (first 25 configs)"):
            try:
                tmp_grid = tmp_dir / f"grid_preview_{_now_slug()}.jsonl"
                base_path = _write_baseline_json(
                    tmp_dir,
                    strategy_name=st.session_state.get("new.strategy_name", "dca_swing"),
                    side="long",
                    params=st.session_state.get("new.baseline_params", {}),
                )
                grid_cmd: List[str] = [
                    PY,
                    st.session_state["new.grid_script"],
                    "--out",
                    str(tmp_grid),
                    "--n",
                    str(max(0, max(25, min(200, n)) - 1)),
                    "--seed",
                    str(seed),
                ]
                if not mode.startswith("Stress test"):
                    grid_cmd += ["--logic-frac", str(float(st.session_state.get("new.logic_frac", 0.35)))]
                if mode.startswith("Stress test"):
                    grid_cmd += [
                        "--mode",
                        "neighborhood",
                        "--base",
                        str(base_path),
                        "--width",
                        str(width),
                        "--vary",
                        ",".join(vary),
                    ]
                else:
                    grid_cmd += ["--mode", "random"]

                _run_cmd(grid_cmd, cwd=REPO_ROOT, label="Generate preview grid")
                _ensure_grid_has_baseline(tmp_grid, base_path, total_n=max(25, min(200, n)))
                rows = _load_jsonl(tmp_grid)[:25]
                st.json(rows[:5])
                st.caption(f"Preview grid rows: {len(rows)} (showing first 5)")
            except Exception as e:
                st.error(str(e))

        # Persist grid settings
        st.session_state["new.grid_width"] = width
        st.session_state["new.grid_n"] = n
        st.session_state["new.grid_seed"] = seed
        st.session_state["new.grid_vary"] = vary
        st.session_state["new.grid_mode2"] = mode
        st.session_state["new.logic_frac"] = float(st.session_state.get("new.logic_frac", 0.35))

        colL, colR = st.columns(2)
        with colL:
            if st.button("← Back"):
                st.session_state["new.step"] = 1
                st.rerun()
        with colR:
            if st.button("Next →", type="primary"):
                st.session_state["new.step"] = 3
                st.rerun()

    # -------------------------------------------------------------------------
    # Step 3: Run batch
    # -------------------------------------------------------------------------
    elif step == 3:
        st.write("Run the batch sweep + rerun, then rank results.")

        data_path = Path(st.session_state.get("new.data_path", ""))
        if not data_path.exists():
            st.error("Missing dataset. Go back to Step 1.")
            st.stop()

        # Run name
        default_name = f"batch_{_now_slug()}_{st.session_state.get('new.strategy_name','strategy')}_{_slug(data_path.stem)}"
        run_name = st.text_input("Run name (folder)", value=default_name, key="new.run_name")

        st.subheader("Compute settings")
        colA, colB, colC = st.columns(3)
        with colA:
            jobs = int(st.number_input("Parallel sweep jobs", min_value=1, max_value=64, value=4, step=1, key="new.jobs"))
        with colB:
            rerun_n = int(st.number_input("Rerun N (full pass)", min_value=1, max_value=10000, value=300, step=10, key="new.rerun_n"))
        with colC:
            top_k = int(st.number_input("Save top-K artifacts (charts)", min_value=0, max_value=500, value=50, step=10, key="new.top_k"))

        st.subheader("Batch gates (keep these permissive for DCA)")
        colG1, colG2, colG3, colG4 = st.columns(4)
        with colG1:
            min_trades = int(st.number_input("Min trades", min_value=0, max_value=10_000, value=0, step=1, key="new.min_trades"))
        with colG2:
            max_fee_impact = float(st.number_input("Max fee impact %", min_value=0.0, max_value=10_000.0, value=250.0, step=10.0, key="new.max_fee"))
        with colG3:
            max_best_over_wins = float(st.number_input("Max best trade / wins", min_value=0.0, max_value=10.0, value=0.95, step=0.01, key="new.max_best"))
        with colG4:
            starting_equity = float(st.number_input("Starting equity", min_value=10.0, max_value=1_000_000.0, value=1000.0, step=100.0, key="new.starting_eq"))

        st.subheader("Ranking")
        score = st.selectbox(
            "Score definition",
            options=["calmar_equity", "profit_dd", "twr_dd", "profit"],
            index=0,
            key="new.score",
        )
        max_dd_filter = st.slider("Optional filter: max drawdown (0 disables)", 0.0, 0.99, 0.0, 0.01, key="new.max_dd_filter")


        st.subheader("Optional: run robustness tests after Batch")
        st.caption("These run *after* Batch completes, using the survivors from this run.")
        colT1, colT2 = st.columns(2)
        with colT1:
            new_do_rs = st.checkbox("Rolling Starts (fragility)", value=False, key="new.do_rs")
            st.caption("Checks if results depend on lucky start dates.")
        with colT2:
            new_do_wf = st.checkbox("Walkforward (discipline)", value=False, key="new.do_wf")
            st.caption("Checks if it survives windowed time splits.")

        # Rough bars/day hint for defaults
        bars_per_day_hint = 1
        try:
            bar_ms_hint = _infer_bar_ms_from_csv(data_path)
            if bar_ms_hint:
                bars_per_day_hint = int(max(1, round(86_400_000 / float(bar_ms_hint))))
        except Exception:
            bars_per_day_hint = 1

        if new_do_rs:
            with st.expander("Rolling Starts settings", expanded=True):
                preset = st.selectbox("Preset", ["Quick", "Standard", "Thorough"], index=1, key="new.rs.preset")
                preset_prev = st.session_state.get("new.rs.preset_prev", None)
                if preset != preset_prev:
                    if preset == "Quick":
                        step_days, min_days = 14, 180
                    elif preset == "Thorough":
                        step_days, min_days = 7, 365
                    else:
                        step_days, min_days = 10, 270
                    st.session_state["new.rs.start_step"] = int(max(1, round(step_days * bars_per_day_hint)))
                    st.session_state["new.rs.min_bars"] = int(max(30, round(min_days * bars_per_day_hint)))
                    st.session_state["new.rs.preset_prev"] = preset

                st.number_input(
                    "Start step (bars)",
                    1,
                    500000,
                    int(st.session_state.get("new.rs.start_step", max(1, int(round(7 * bars_per_day_hint))))),
                    5,
                    key="new.rs.start_step",
                )
                st.number_input(
                    "Min bars per start",
                    30,
                    5000000,
                    int(st.session_state.get("new.rs.min_bars", max(30, int(round(365 * bars_per_day_hint))))),
                    10,
                    key="new.rs.min_bars",
                )

        if new_do_wf:
            with st.expander("Walkforward settings", expanded=True):
                preset = st.selectbox("Preset", ["Quick", "Standard", "Thorough"], index=1, key="new.wf.preset")
                preset_prev = st.session_state.get("new.wf.preset_prev", None)
                if preset != preset_prev:
                    if preset == "Quick":
                        window_days, step_days = 30, 15
                    elif preset == "Thorough":
                        window_days, step_days = 180, 30
                    else:
                        window_days, step_days = 90, 30
                    st.session_state["new.wf.window_days"] = int(window_days)
                    st.session_state["new.wf.step_days"] = int(step_days)
                    st.session_state["new.wf.preset_prev"] = preset

                st.number_input("Window (days)", min_value=1, max_value=3650, step=1, key="new.wf.window_days")
                st.number_input("Step (days)", min_value=1, max_value=3650, step=1, key="new.wf.step_days")
                # We'll clamp min-bars at run-time once bars/day is known from the run meta
                st.number_input("Min bars per window", min_value=1, max_value=5_000_000, step=1, key="new.wf.min_bars")
                st.number_input("Jobs", min_value=1, max_value=64, step=1, value=8, key="new.wf.jobs")
        do_run = st.button("Run batch stress test", type="primary")

        if do_run:
            try:
                t0 = time.time()
                tmp_run_dir = tmp_dir / f"run_{_now_slug()}"
                tmp_run_dir.mkdir(parents=True, exist_ok=True)

                # UI: unified run monitor (Sprint 3)
                st.subheader("Run monitor")
                grid_comp_ph = st.empty()
                stages: List[_PipelineStage] = [
                    _PipelineStage("grid", "Variants"),
                    _PipelineStage("batch", "Batch"),
                    _PipelineStage("post", "Postprocess"),
                ]
                if bool(st.session_state.get("new.do_rs", False)):
                    stages.append(_PipelineStage("rs", "Rolling Starts"))
                if bool(st.session_state.get("new.do_wf", False)):
                    stages.append(_PipelineStage("wf", "Walkforward"))
                pipe = _PipelineUI(stages)

                # 1) Write baseline
                base_path = _write_baseline_json(
                    tmp_run_dir,
                    strategy_name=st.session_state.get("new.strategy_name", "dca_swing"),
                    side="long",
                    params=st.session_state.get("new.baseline_params", {}),
                )

                # 2) Generate grid
                grid_path = tmp_run_dir / f"grid_{st.session_state['new.grid_n']}_seed{st.session_state['new.grid_seed']}.jsonl"
                grid_cmd: List[str] = [
                    PY,
                    st.session_state["new.grid_script"],
                    "--out",
                    str(grid_path),
                    "--n",
                    str(max(0, int(st.session_state["new.grid_n"]) - 1)),
                    "--seed",
                    str(int(st.session_state["new.grid_seed"])),
                ]
                if not str(st.session_state.get("new.grid_mode2", "")).startswith("Stress test"):
                    grid_cmd += ["--logic-frac", str(float(st.session_state.get("new.logic_frac", 0.35)))]
                if str(st.session_state.get("new.grid_mode2", "")).startswith("Stress test"):
                    grid_cmd += [
                        "--mode",
                        "neighborhood",
                        "--base",
                        str(base_path),
                        "--width",
                        str(st.session_state.get("new.grid_width", "medium")),
                        "--vary",
                        ",".join(st.session_state.get("new.grid_vary", ["deposits","buys"])),
                    ]
                else:
                    grid_cmd += ["--mode", "random"]

                pipe.run("grid", grid_cmd, cwd=REPO_ROOT)
                _ensure_grid_has_baseline(grid_path, base_path, total_n=int(st.session_state["new.grid_n"]))

                # Grid composition (dopamine loop): show + save
                try:
                    run_dir = RUNS_DIR / str(run_name)
                    comp = _summarize_grid_composition(Path(str(grid_path)))
                    _write_json(run_dir / "grid_meta.json", comp)
                    with grid_comp_ph.container():
                        _render_grid_composition(comp)
                except Exception as _e:
                    # Never fail the run on a summary widget
                    with grid_comp_ph.container():
                        st.caption(f"Grid composition unavailable: {_e}")

                # 3) Batch
                template_path = str(st.session_state.get("new.template_path", "strategies.dca_swing:Strategy"))
                market_mode = str(st.session_state.get("new.market_mode", "spot"))
                batch_cmd: List[str] = [
                    PY,
                    "-m",
                    "engine.batch",
                    "--data",
                    str(data_path),
                    "--grid",
                    str(grid_path),
                    "--template",
                    template_path,
                    "--market-mode",
                    market_mode,
                    "--run-name",
                    str(run_name),
                    "--out",
                    str(RUNS_DIR),
                    "--jobs",
                    str(jobs),
                    "--fast-sweep",
                    "--min-trades",
                    str(min_trades),
                    "--max-fee-impact-pct",
                    str(max_fee_impact),
                    "--max-best-over-wins",
                    str(max_best_over_wins),
                    "--sweep-sort-by",
                    "equity.net_profit_ex_cashflows",
                    "--sweep-sort-desc",
                    "--sort-by",
                    "equity.net_profit_ex_cashflows",
                    "--sort-desc",
                    "--rerun-n",
                    str(rerun_n),
                    "--top-k",
                    str(top_k),
                    "--starting-equity",
                    str(starting_equity),
                ]
                batch_progress = RUNS_DIR / str(run_name) / "progress" / "batch.jsonl"
                batch_progress.parent.mkdir(parents=True, exist_ok=True)
                # Persist enough metadata for replay tools (grid/data/template/etc.)
                run_dir = RUNS_DIR / str(run_name)
                meta_path = run_dir / "batch_meta.json"
                meta: Dict[str, Any] = {}
                try:
                    if meta_path.exists():
                        meta = _read_json(meta_path)  # type: ignore
                except Exception:
                    meta = {}

                # Estimate bars/day from the dataset so downstream Rolling Starts / Walkforward
                # interpret "min bars" correctly. Falls back to 1 (daily) if inference fails.
                bar_ms = _infer_bar_ms_from_csv(Path(str(data_path)))
                if bar_ms and bar_ms > 0:
                    bars_per_day = int(max(1, round(86_400_000 / float(bar_ms))))
                else:
                    bars_per_day = 1
                meta.update({
                    "run_name": str(run_name),
                    "grid_path": str(grid_path),
                    "data_path": str(data_path),
                    "template": str(template_path),
                    "market_mode": market_mode,
                    "bars_per_day": int(bars_per_day),
                    "ui_written_at": time.time(),
                })
                _write_json(meta_path, meta)
                batch_cmd += ["--no-progress", "--progress-file", str(batch_progress), "--progress-every", "25"]
                pipe.run("batch", batch_cmd, cwd=REPO_ROOT, progress_path=batch_progress.parent)

                run_dir = RUNS_DIR / str(run_name)
                if not run_dir.exists():
                    # fallback: find latest
                    runs2 = _list_runs()
                    run_dir = runs2[0] if runs2 else run_dir

                # 4) Postprocess ranking
                post_cmd: List[str] = [
                    PY,
                    str(REPO_ROOT / "tools" / "postprocess_batch_results.py"),
                    "--from-run",
                    str(run_dir),
                    "--score",
                    str(score),
                    "--top-n",
                    "200",
                ]
                if max_dd_filter and float(max_dd_filter) > 0:
                    post_cmd += ["--max-dd", str(float(max_dd_filter))]

                pipe.run("post", post_cmd, cwd=REPO_ROOT)

                # 5) Optional: run Rolling Starts / Walkforward immediately after Batch
                post_ok = True
                try:
                    do_rs = bool(st.session_state.get("new.do_rs", False))
                    do_wf = bool(st.session_state.get("new.do_wf", False))
                    if do_rs or do_wf:
                        st.info("Running selected robustness tests…")

                        frames2 = load_batch_frames(run_dir)
                        survivors2, _src2 = pick_survivors(frames2)
                        survivors_ids = survivors2["config_id"].astype(str).tolist()
                        N = len(survivors_ids)

                        ids_file = run_dir / "post" / "survivor_ids.txt"
                        ids_file.parent.mkdir(parents=True, exist_ok=True)
                        ids_file.write_text("\n".join(survivors_ids) + "\n", encoding="utf-8")

                        bars_per_day = _bars_per_day_from_run_meta(run_dir)

                        rs_root = run_dir / "rolling_starts"
                        wf_root = run_dir / "walkforward"

                        if do_rs and N > 0:
                            start_step = int(st.session_state.get("new.rs.start_step", max(1, int(round(10 * bars_per_day)))))
                            min_bars = int(st.session_state.get("new.rs.min_bars", max(30, int(round(270 * bars_per_day)))))

                            rs_out_dir = rs_root / f"rs_step{start_step}_min{min_bars}_n{N}"
                            rs_progress = rs_out_dir / "progress" / "rolling_starts.jsonl"
                            rs_progress.parent.mkdir(parents=True, exist_ok=True)

                            cmd = [
                                PY,
                                "-m",
                                "research.rolling_starts",
                                "--from-run",
                                str(run_dir),
                                "--out",
                                str(rs_out_dir),
                                "--ids",
                                str(ids_file),                                "--top-n",
                                str(N),
                                "--start-step",
                                str(start_step),
                                "--min-bars",
                                str(min_bars),
                                "--seed",
                                "1",
                                "--starting-equity",
                                str(float(st.session_state.get("new.starting_eq", 1000.0) or 1000.0)),
                                "--jobs", "8",
                                "--no-progress",
                                "--progress-file",
                                str(rs_progress),
                                "--progress-every",
                                "10",
                            ]
                            pipe.run("rs", cmd, cwd=REPO_ROOT, progress_path=rs_progress.parent)

                        if do_wf and N > 0:
                            window_days = int(st.session_state.get("new.wf.window_days", 90))
                            step_days = int(st.session_state.get("new.wf.step_days", 30))
                            min_bars = int(st.session_state.get("new.wf.min_bars", 1))
                            expected_window_bars = int(max(1, round(window_days * bars_per_day)))
                            min_bars_effective = int(min(int(min_bars), int(expected_window_bars)))
                            jobs = int(st.session_state.get("new.wf.jobs", 8))

                            wf_out_dir = wf_root / f"wf_win{window_days}_step{step_days}_min{min_bars_effective}_n{N}"
                            wf_progress = wf_out_dir / "progress" / "walkforward.jsonl"
                            wf_progress.parent.mkdir(parents=True, exist_ok=True)

                            cmd = [
                                PY,
                                "-m",
                                "engine.walkforward",
                                "--from-run",
                                str(run_dir),
                                "--out",
                                str(wf_out_dir),                                "--top-n",
                                str(N),
                                "--window-days",
                                str(window_days),
                                "--step-days",
                                str(step_days),
                                "--min-bars",
                                str(min_bars_effective),
                                "--seed",
                                "1",
                                "--starting-equity",
                                str(float(st.session_state.get("new.starting_eq", 1000.0) or 1000.0)),
                                "--jobs",
                                str(jobs),
                                "--no-progress",
                                "--progress-file",
                                str(wf_progress),
                                "--progress-every",
                                "10",
                            ]
                            pipe.run("wf", cmd, cwd=REPO_ROOT, progress_path=wf_progress.parent)

                except Exception as e:
                    post_ok = False
                    st.warning(f"Post-batch tests failed: {e}")


                # Trust layer: write/refresh manifest.json for this run
                try:
                    _ensure_manifest(run_dir)
                except Exception:
                    pass

                st.success(f"Done in {time.time()-t0:.1f}s. Run saved to: {run_dir.name}")

                # Switch to opening this run (so Results can find it)
                st.session_state["selected_run"] = run_dir.name
                st.session_state["ui.open_run_next"] = run_dir.name  # set on next rerun before widget instantiates

                # Reset wizard
                st.session_state["new.step"] = 0

                if post_ok:
                    # After a successful full run, jump to Results by default
                    st.session_state["ui.section_next"] = "2) Results & Autopsy"
                    st.session_state.setdefault("ui.stage", "batch")
                    st.rerun()
                else:
                    st.info("Some selected robustness tests failed. Review the logs above, then open the run in Results when ready.")
                    if st.button("Open Results & Autopsy", key="run.goto_results_after_fail"):
                        st.session_state["ui.section_next"] = "2) Results & Autopsy"
                        st.session_state.setdefault("ui.stage", "batch")
                        st.rerun()

            except Exception as e:
                st.error(str(e))
                st.stop()

    st.stop()

# =============================================================================
# Existing run analysis
# =============================================================================

selected_run_name = st.session_state.get("selected_run", "")
if not selected_run_name:
    st.info("No runs found yet. Create a new run from the sidebar.")
    st.stop()

run_dir = RUNS_DIR / selected_run_name
if not run_dir.exists():
    st.error(f"Run folder not found: {run_dir}")
    st.stop()

st.subheader(f"Run: {selected_run_name}")
meta_path = run_dir / "batch_meta.json"
meta = _read_json(meta_path)
if not meta and not meta_path.exists():
    st.warning("This run is missing batch_meta.json (likely an interrupted run). You can still view any results that were written.")
if meta:
    st.caption(f"Data: {meta.get('data','?')}  |  Grid: {meta.get('grid','?')}  |  Template: {meta.get('template','?')}")

frames = load_batch_frames(run_dir)
survivors, survivor_source = pick_survivors(frames)

if survivors.empty:
    st.warning("No results found in this run folder (or everything is empty).")
    st.stop()

# Ranked frame (if present) gives score columns
ranked = frames.get("ranked")
if ranked is not None and not ranked.empty and "config_id" in ranked.columns:
    # Keep ranked order, but ensure gates.passed=True if available
    ranked2 = ranked.copy()
    if "gates.passed" in ranked2.columns:
        ranked2 = ranked2[ranked2["gates.passed"].astype(bool)].copy()
    survivors = survivors.drop(columns=[c for c in survivors.columns if c.startswith("score.")], errors="ignore")
    survivors = survivors.merge(
        ranked2[["config_id"] + [c for c in ranked2.columns if c.startswith("score.") or c.startswith("pareto.")]],
        how="left",
        on="config_id",
    )

top_map = _parse_top_artifact_dirs(run_dir)

# Stage directories
rs_root = run_dir / "rolling_starts"
wf_root = run_dir / "walkforward"

rs_latest = _pick_latest_dir(rs_root, "rs_*") if rs_root.exists() else None
wf_latest = _pick_latest_dir(wf_root, "wf_*") if wf_root.exists() else None

# =============================================================================
# MVP navigation: Build & Run vs Results
# =============================================================================

section_pick = str(st.session_state.get("ui.section", "2) Results & Autopsy"))

if section_pick.startswith("1)"):
    st.header("Build & Run")

    st.caption("Run additional stress tests on the *current* run’s survivors. Results live in the Results & Autopsy section.")
    st.write(f"Survivors detected: **{len(survivors):,}** (source: **{survivor_source}**).")

    bars_per_day = _bars_per_day_from_run_meta(run_dir)

    # ---- Test selection
    st.subheader("Choose tests to run")
    col_a, col_b = st.columns(2)
    with col_a:
        do_rs = st.checkbox("Rolling Starts (start-date fragility)", value=False, key="runner.do_rs")
        st.caption("Same strategy, many start dates → reveals ‘lucky start’ dependence.")
    with col_b:
        do_wf = st.checkbox("Walkforward (out-of-sample-ish windows)", value=False, key="runner.do_wf")
        st.caption("Repeats training/testing through time → reveals generalization vs overfit.")

    # ---- Config panels
    rs_out_dir = None
    wf_out_dir = None
    ids_file = None

    survivors_ids = survivors["config_id"].astype(str).tolist()
    N = len(survivors_ids)

    # Persist survivor ids for reproducible replays
    ids_file = run_dir / "post" / "survivor_ids.txt"
    ids_file.parent.mkdir(parents=True, exist_ok=True)
    ids_file.write_text("\n".join(survivors_ids) + "\n", encoding="utf-8")

    if do_rs:
        st.markdown("#### Rolling Starts settings")
        with st.expander("Rolling Starts settings", expanded=True):
            preset = st.selectbox("Preset", ["Quick", "Standard", "Thorough"], index=1, key="rs.preset")
            # Apply preset defaults once per change (mirrors RS page behavior)
            preset_prev = st.session_state.get("rs.preset_prev", None)
            if preset != preset_prev:
                if preset == "Quick":
                    step_days, min_days = 14, 180
                elif preset == "Thorough":
                    step_days, min_days = 7, 365
                else:
                    step_days, min_days = 10, 270
                st.session_state["rs.start_step"] = int(max(1, round(step_days * bars_per_day)))
                st.session_state["rs.min_bars"] = int(max(30, round(min_days * bars_per_day)))
                st.session_state["rs.preset_prev"] = preset

            start_step = int(
                st.number_input(
                    "Start step (bars)",
                    1,
                    500000,
                    int(st.session_state.get("rs.start_step", max(1, int(round(7 * bars_per_day))))),
                    5,
                    key="rs.start_step",
                )
            )
            min_bars = int(
                st.number_input(
                    "Min bars per start",
                    30,
                    5000000,
                    int(st.session_state.get("rs.min_bars", max(30, int(round(365 * bars_per_day))))),
                    10,
                    key="rs.min_bars",
                )
            )

            rs_out_dir = rs_root / f"rs_step{start_step}_min{min_bars}_n{N}"
            st.caption(f"Output: {rs_out_dir}")

    if do_wf:
        st.markdown("#### Walkforward settings")
        with st.expander("Walkforward settings", expanded=True):
            preset = st.selectbox("Preset", ["Quick", "Standard", "Thorough"], index=1, key="wf.preset")
            preset_prev = st.session_state.get("wf.preset_prev", None)
            if preset != preset_prev:
                if preset == "Quick":
                    window_days, step_days = 30, 15
                elif preset == "Thorough":
                    window_days, step_days = 180, 30
                else:
                    window_days, step_days = 90, 30
                st.session_state["wf.window_days"] = int(window_days)
                st.session_state["wf.step_days"] = int(step_days)
                st.session_state["wf.preset_prev"] = preset

            window_days = int(st.number_input("Window (days)", min_value=1, max_value=3650, step=1, key="wf.window_days"))
            step_days = int(st.number_input("Step (days)", min_value=1, max_value=3650, step=1, key="wf.step_days"))

            expected_window_bars = int(max(1, round(window_days * bars_per_day)))
            st.caption(f"Expected bars per window: ~{expected_window_bars:,}. (Min bars must be ≤ this.)")

            max_mb = int(max(1, expected_window_bars))
            if "wf.min_bars" not in st.session_state:
                st.session_state["wf.min_bars"] = int(max_mb)
            if int(st.session_state.get("wf.min_bars", 1)) > int(max_mb):
                st.session_state["wf.min_bars"] = int(max_mb)

            min_bars = int(st.number_input(
                "Min bars per window",
                min_value=1,
                max_value=max_mb,
                step=1,
                key="wf.min_bars",
            ))
            if "wf.jobs" not in st.session_state:
                st.session_state["wf.jobs"] = 8
            jobs = int(st.number_input("Jobs", min_value=1, max_value=64, step=1, key="wf.jobs"))

            min_bars_effective = int(min(int(min_bars), int(expected_window_bars)))
            if int(min_bars) != int(min_bars_effective):
                st.warning(
                    f"Min bars ({min_bars}) exceeds expected bars/window (~{expected_window_bars}). "
                    f"Will clamp to {min_bars_effective}."
                )

            wf_out_dir = wf_root / f"wf_win{window_days}_step{step_days}_min{min_bars_effective}_n{N}"
            st.caption(f"Output: {wf_out_dir}")

    st.divider()

    run_btn = st.button("Run selected tests", type="primary", disabled=(not do_rs and not do_wf))
    if run_btn:
        try:
            st.subheader("Run monitor")
            stages: List[_PipelineStage] = []
            if do_rs and rs_out_dir is not None:
                stages.append(_PipelineStage("rs", "Rolling Starts"))
            if do_wf and wf_out_dir is not None:
                stages.append(_PipelineStage("wf", "Walkforward"))
            pipe = _PipelineUI(stages)

            if do_rs and rs_out_dir is not None:
                rs_progress = rs_out_dir / "progress" / "rolling_starts.jsonl"
                rs_progress.parent.mkdir(parents=True, exist_ok=True)
                cmd = [
                    PY,
                    "-m",
                    "research.rolling_starts",
                    "--from-run",
                    str(run_dir),
                    "--out",
                    str(rs_out_dir),
                    "--ids",
                    str(ids_file),
                    "--top-n",
                    str(N),
                    "--start-step",
                    str(int(st.session_state.get("rs.start_step", 1))),
                    "--min-bars",
                    str(int(st.session_state.get("rs.min_bars", 30))),
                    "--seed",
                    "1",
                    "--starting-equity",
                    str(float(meta.get("starting_equity", 1000.0) or 1000.0)),
                    "--jobs", "8",
                    "--no-progress",
                    "--progress-file",
                    str(rs_progress),
                    "--progress-every",
                    "10",
                ]
                pipe.run("rs", cmd, cwd=REPO_ROOT, progress_path=rs_progress.parent)

            if do_wf and wf_out_dir is not None:
                wf_progress = wf_out_dir / "progress" / "walkforward.jsonl"
                wf_progress.parent.mkdir(parents=True, exist_ok=True)
                window_days = int(st.session_state.get("wf.window_days", 90))
                step_days = int(st.session_state.get("wf.step_days", 30))
                min_bars = int(st.session_state.get("wf.min_bars", 1))
                expected_window_bars = int(max(1, round(window_days * bars_per_day)))
                min_bars_effective = int(min(int(min_bars), int(expected_window_bars)))
                cmd = [
                    PY,
                    "-m",
                    "engine.walkforward",
                    "--from-run",
                    str(run_dir),
                    "--out",
                    str(wf_out_dir),
                    "--top-n",
                    str(N),
                    "--window-days",
                    str(window_days),
                    "--step-days",
                    str(step_days),
                    "--min-bars",
                    str(min_bars_effective),
                    "--seed",
                    "1",
                    "--starting-equity",
                    str(float(meta.get("starting_equity", 1000.0) or 1000.0)),
                    "--jobs",
                    str(int(st.session_state.get("wf.jobs", 8))),
                    "--no-progress",
                    "--progress-file",
                    str(wf_progress),
                    "--progress-every",
                    "10",
                ]
                pipe.run("wf", cmd, cwd=REPO_ROOT, progress_path=wf_progress.parent)

            st.success("Selected tests completed.")

            # Trust layer: refresh manifest now that RS/WF evidence may have changed
            try:
                _ensure_manifest(run_dir)
            except Exception:
                pass
            st.session_state["ui.section_next"] = "2) Results & Autopsy"
            if do_wf:
                st.session_state["ui.stage"] = "wf"
            elif do_rs:
                st.session_state["ui.stage"] = "rs"
            else:
                st.session_state["ui.stage"] = "grand"
            st.rerun()

        except Exception as e:
            st.error(str(e))
            st.stop()

    st.stop()

# ---- Results section
st.header("Results & Autopsy")

# -----------------------------------------------------------------------------
# Run status strip (MVP contract: Results is view-only)
# -----------------------------------------------------------------------------
total_cfg = int(len(survivors)) if survivors is not None else 0

# Batch is "ready" if we have a run loaded.
batch_icon = "✅"
batch_label = "ready"

# Rolling Starts coverage
rs_done = 0
try:
    if rs_latest is not None:
        _rs_sum_status = load_rs_summary(run_dir, rs_latest)
        if _rs_sum_status is not None and not _rs_sum_status.empty and "config_id" in _rs_sum_status.columns:
            rs_done = int(_rs_sum_status["config_id"].astype(str).nunique())
except Exception:
    rs_done = 0

if rs_done <= 0:
    rs_icon = "⚠️"
    rs_label = "missing"
elif total_cfg > 0 and rs_done < total_cfg:
    rs_icon = "⚠️"
    rs_label = f"partial ({rs_done}/{total_cfg})"
else:
    rs_icon = "✅"
    rs_label = "ready"

# Walkforward coverage
wf_done = 0
try:
    if wf_latest is not None:
        _wf_sum_status = load_wf_summary(wf_latest)
        if _wf_sum_status is not None and not _wf_sum_status.empty and "config_id" in _wf_sum_status.columns:
            wf_done = int(_wf_sum_status["config_id"].astype(str).nunique())
except Exception:
    wf_done = 0

if wf_done <= 0:
    wf_icon = "⚠️"
    wf_label = "missing"
elif total_cfg > 0 and wf_done < total_cfg:
    wf_icon = "⚠️"
    wf_label = f"partial ({wf_done}/{total_cfg})"
else:
    wf_icon = "✅"
    wf_label = "ready"

missing_rs = (rs_done <= 0) or (total_cfg > 0 and rs_done < total_cfg)
missing_wf = (wf_done <= 0) or (total_cfg > 0 and wf_done < total_cfg)

strip1, strip2, strip3, strip4 = st.columns([1.1, 1.5, 1.2, 1.6])
with strip1:
    st.markdown(f"**Batch:** {batch_icon} {batch_label}")
with strip2:
    st.markdown(f"**Rolling Starts:** {rs_icon} {rs_label}")
with strip3:
    st.markdown(f"**Walkforward:** {wf_icon} {wf_label}")
with strip4:
    if missing_rs or missing_wf:
        if st.button("Go run missing tests", type="primary", key="results.go_run_missing"):
            st.session_state["runner.do_rs"] = bool(missing_rs)
            st.session_state["runner.do_wf"] = bool(missing_wf)
            st.session_state["ui.section_next"] = "1) Build & Run"
            st.rerun()
    else:
        st.caption("Results is view-only. Run tests from Build & Run.")


# -----------------------------------------------------------------------------
# Trust & comparability (Sprint 4/5)
# -----------------------------------------------------------------------------
manifest = {}
try:
    manifest = _ensure_manifest(run_dir)
except Exception:
    manifest = {}

warns = []
try:
    warns = _comparability_warnings(manifest)
except Exception:
    warns = []

# Optional cross-run compare (Sprint 5)
compare_warns: List[str] = []
compare_manifest: Dict[str, Any] = {}
compare_run = None
try:
    runs_root = REPO_ROOT / "runs"
    other_runs = [p.name for p in runs_root.glob("batch_*") if p.is_dir() and p.name != run_dir.name]
    other_runs = sorted(other_runs, reverse=True)
except Exception:
    other_runs = []

with st.container(border=True):
    st.markdown("#### Trust & comparability")
    ds = (manifest.get("dataset") or {}) if isinstance(manifest, dict) else {}
    ds_path = ds.get("path_abs") or ds.get("path")
    fp = (ds.get("fingerprint") or {}) if isinstance(ds, dict) else {}
    git_head = (manifest.get("engine_git_head") if isinstance(manifest, dict) else None) or "unknown"

    cA, cB, cC = st.columns([1.4, 1.2, 1.2])
    with cA:
        st.caption("Dataset")
        st.code(str(ds_path or "unknown"), language="text")
    with cB:
        st.caption("Fingerprint")
        dig = str(fp.get("digest") or "")
        st.code((dig[:10] + "…" + dig[-10:]) if dig else "unknown", language="text")
    with cC:
        st.caption("Engine git")
        gh = str(git_head or "")
        st.code((gh[:10] + "…" + gh[-6:]) if gh and gh != "unknown" else "unknown", language="text")

    if other_runs:
        compare_run = st.selectbox("Compare to another run (optional)", ["(none)"] + other_runs, index=0, key="trust.compare_run")
        if compare_run and compare_run != "(none)":
            try:
                compare_manifest = _ensure_manifest(runs_root / compare_run)
                compare_warns = _compare_manifests(manifest if isinstance(manifest, dict) else {}, compare_manifest if isinstance(compare_manifest, dict) else {})
            except Exception:
                compare_warns = ["Could not load/compare the selected run manifest."]

    if warns:
        for w in warns:
            st.warning(w)
    else:
        st.success("Manifest present. Dataset + test parameters look comparable.")

    if compare_run and compare_run != "(none)":
        if compare_warns:
            st.info("Cross-run comparability warnings:")
            for w in compare_warns:
                st.warning(w)
        else:
            st.success("Selected run appears comparable (dataset + test parameters match).")

    try:
        st.download_button(
            "Download manifest.json",
            data=json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name=f"{selected_run_name}_manifest.json",
            key="dl.manifest",
        )
    except Exception:
        pass

    with st.expander("Show manifest (raw JSON)", expanded=False):
        st.json(manifest)


# Force cockpit mode: show the unified Grand Verdict cockpit only.
st.session_state["ui.stage"] = "grand"
stage_pick = "grand"

st.divider()
# =============================================================================
# Stage A: Batch
# =============================================================================

if stage_pick == "batch":
    st.write("### A) Batch results (sweep + rerun)")
    st.caption(f"Survivor source: **{survivor_source}**. (Increase rerun_n if you want more survivors in full_passed.)")

    # Questions
    with st.expander("Batch questions (filters)", expanded=True):
        batch_ans = _question_ui(batch_questions(), key_prefix="q.batch")
    dfA = apply_stage_eval(survivors, stage_key="batch", questions=batch_questions(), answers=batch_ans)

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        show_pass = st.checkbox("Show PASS", value=True, key="f.batch.pass")
    with col2:
        show_warn = st.checkbox("Show WARN", value=True, key="f.batch.warn")
    with col3:
        show_fail = st.checkbox("Show FAIL", value=False, key="f.batch.fail")

    keep = []
    if show_pass:
        keep.append("PASS")
    if show_warn:
        keep.append("WARN")
    if show_fail:
        keep.append("FAIL")

    df_show = dfA[dfA["batch.verdict"].isin(keep)].copy()

    # ---------------------------------------------------------------------
    # Visual scan (turn the table into something a human can reason about)
    # ---------------------------------------------------------------------
    st.write("#### Quick visual scan")

    id_col = "config_id"
    label_col = _pick_col(df_show, ["config.label", "label", "config_label"])
    verdict_col = _pick_col(df_show, ["batch.verdict", "verdict", "batch_verdict"])

    profit_col = _pick_col(df_show, ["equity.net_profit_ex_cashflows", "equity.net_profit", "equity.net_profit_ex_cashflow"])
    dd_col = _pick_col(df_show, ["performance.max_drawdown_equity", "performance.max_drawdown", "equity.max_drawdown"])
    trades_col = _pick_col(df_show, ["trades_summary.trades_closed", "trades.closed", "trades_closed", "trades"])

    calmar_col = _pick_col(df_show, ["score.calmar_equity", "performance.calmar", "calmar"])

    if profit_col and dd_col and px is not None and go is not None and not df_show.empty:
        plot_df = df_show.copy()
        plot_df["_profit"] = _to_float_series(plot_df[profit_col])
        plot_df["_dd"] = _drawdown_to_frac(plot_df[dd_col])
        if trades_col:
            plot_df["_trades"] = _to_float_series(plot_df[trades_col]).fillna(0.0)
        else:
            plot_df["_trades"] = 0.0

        plot_df["_label"] = plot_df[label_col].astype(str) if label_col else ""
        plot_df["_verdict"] = plot_df[verdict_col].astype(str) if verdict_col else "?"

        # Frontier hygiene: by default, ignore ultra-low-activity configs so the frontier is meaningful.
        # (Otherwise you often get near-zero drawdown "do nothing" configs anchoring the left edge.)
        max_tr = int(max(0.0, float(pd.to_numeric(plot_df["_trades"], errors="coerce").max() or 0.0)))
        max_slider = max(0, min(200, max_tr))
        default_tr = 3 if max_slider >= 3 else max_slider
        frontier_min_trades = st.slider(
            "Pareto frontier: min trades",
            min_value=0,
            max_value=max_slider,
            value=default_tr,
            step=1,
            help="Frontier is shown only among configs with at least this many trades (to avoid 'do nothing' near-zero DD anchors).",
        )

        frontier_df = plot_df
        if frontier_min_trades > 0:
            frontier_df = plot_df[plot_df["_trades"] >= frontier_min_trades].copy()

        # Scatter: profit vs drawdown
        fig = px.scatter(
            plot_df,
            x="_dd",
            y="_profit",
            color="_verdict",
            size="_trades",
            hover_data={
                id_col: True,
                "_label": True,
                "_profit": ":.2f",
                "_dd": ":.4f",
                "_trades": ":.0f",
            },
            render_mode="webgl",
            title="Return vs max drawdown (each dot is a strategy)",
        )
        fig.update_layout(
            xaxis_title="Max drawdown (fraction, lower is better)",
            yaxis_title="Net profit (excluding deposits)",
            legend_title_text="Batch verdict",
            height=520,
            margin=dict(l=10, r=10, t=50, b=10),
        )

        # Pareto frontier overlay (can't improve profit without worsening drawdown)
        frontier = _pareto_frontier_rows(frontier_df, "_dd", "_profit")
        if not frontier.empty:
            hover_text = None
            if "_label" in frontier.columns and "config_id" in frontier.columns:
                hover_text = frontier["_label"].astype(str) + " • " + frontier["config_id"].astype(str)
            elif "_label" in frontier.columns:
                hover_text = frontier["_label"].astype(str)
            elif "config_id" in frontier.columns:
                hover_text = frontier["config_id"].astype(str)

            fig.add_trace(
                go.Scatter(
                    x=frontier["_dd"],
                    y=frontier["_profit"],
                    mode="lines+markers",
                    name="Pareto frontier",
                    text=hover_text,
                    hovertemplate="%{text}<br>dd=%{x:.4f}<br>profit=%{y:.2f}<extra></extra>",
                    line=dict(width=2),
                )
            )

        _plotly(fig)

        # Quick sanity: list frontier points (helps confirm it's "real" and not plotting artifacts)
        with st.expander("Pareto frontier points", expanded=False):
            if frontier.empty:
                st.caption("No frontier points available (check filters).")
            else:
                show_cols = [c for c in ["config_id", "_label", "_verdict", "_trades", "_dd", "_profit"] if c in frontier.columns]
                st.dataframe(frontier[show_cols].sort_values("_dd", ascending=True), width="stretch")
    else:
        st.info("Scatter plot unavailable (missing Plotly or required columns).")

    # ---------------------------------------------------------------------
    # Top candidates as cards (humans think in tradeoffs, not columns)
    # ---------------------------------------------------------------------
    st.write("#### Top candidates (cards)")

    rank_col = _pick_col(df_show, ["score.profit", profit_col] if profit_col else ["score.profit"])
    if rank_col is None:
        rank_col = profit_col

    cards_df = df_show.copy()
    if rank_col:
        cards_df["_rank"] = _to_float_series(cards_df[rank_col])
        cards_df = cards_df.sort_values("_rank", ascending=False)
    cards_df = cards_df.head(12).copy()

    if cards_df.empty:
        st.info("No rows to show.")
    else:
        cols_cards = st.columns(3)
        for i, (_, r) in enumerate(cards_df.iterrows()):
            cfg_id = str(r.get(id_col, ""))
            label = str(r.get(label_col, cfg_id)) if label_col else cfg_id
            verdict = str(r.get(verdict_col, "")) if verdict_col else ""

            profit_v = float(r.get(profit_col, float("nan"))) if profit_col else float("nan")
            dd_v = float(r.get(dd_col, float("nan"))) if dd_col else float("nan")
            dd_frac = float(_drawdown_to_frac(pd.Series([dd_v])).iloc[0]) if dd_col else float("nan")
            trades_v = int(float(r.get(trades_col, 0))) if trades_col and str(r.get(trades_col, "")).strip() != "" else 0
            calmar_v = float(r.get(calmar_col, float("nan"))) if calmar_col else float("nan")

            with cols_cards[i % 3]:
                with st.container(border=True):
                    st.write(f"**{label}**")
                    st.caption(f"{cfg_id} • {verdict}")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Profit", _fmt_money(profit_v) if profit_col else "n/a")
                    m2.metric("Max DD", _fmt_pct(dd_frac) if dd_col else "n/a")
                    m3.metric("Trades", f"{trades_v}" if trades_col else "n/a")
                    if calmar_col:
                        st.caption(f"Calmar: {_fmt_num(calmar_v, digits=3)}")

                    if st.button("Inspect", key=f"batch.inspect.{cfg_id}"):
                        st.session_state["ui.batch.inspect_id"] = cfg_id
                        st.session_state["ui.batch.scroll_to_inspect"] = True
                        st.rerun()

    # ---------------------------------------------------------------------
    # Heatmap table (percentiles) for quick pattern recognition
    # ---------------------------------------------------------------------
    st.write("#### Quick scan heatmap (scores among shown rows — higher is better)")
    heat_base = df_show.copy()
    n = len(heat_base)

    heat = pd.DataFrame()
    heat["config_id"] = heat_base["config_id"].astype(str).str.strip()
    if label_col:
        heat["label"] = heat_base[label_col].astype(str)
    if verdict_col:
        heat["verdict"] = heat_base[verdict_col].astype(str)

    # Raw helpers
    if profit_col:
        heat["profit"] = _to_float_series(heat_base[profit_col])
        heat["profit_%ile"] = (_goodness_percentile(heat_base[profit_col], low_is_good=False) * 100.0).round(1)
    if dd_col:
        dd_frac = _drawdown_to_frac(heat_base[dd_col])
        heat["max_dd"] = dd_frac
        heat["dd_good_%ile"] = (_goodness_percentile(dd_frac, low_is_good=True) * 100.0).round(1)
    if trades_col:
        heat["trades"] = _to_float_series(heat_base[trades_col]).round(0)
        # More trades generally = more evidence; treat low-is-bad (so low_is_good=False)
        heat["trades_%ile"] = (_goodness_percentile(heat_base[trades_col], low_is_good=False) * 100.0).round(1)
    if calmar_col:
        heat["calmar"] = _to_float_series(heat_base[calmar_col])
        heat["calmar_%ile"] = (_goodness_percentile(heat_base[calmar_col], low_is_good=False) * 100.0).round(1)

    # Display with color on the percentile columns
    pct_cols = [c for c in heat.columns if c.endswith("%ile")]
    disp_cols = ["config_id"] + (["label"] if "label" in heat.columns else []) + (["verdict"] if "verdict" in heat.columns else [])
    disp_cols += [c for c in ["profit", "max_dd", "trades", "calmar"] if c in heat.columns]
    disp_cols += pct_cols

    heat_disp = heat[disp_cols].copy()
    sty = heat_disp.style
    for c in pct_cols:
        sty = sty.background_gradient(subset=[c], cmap="RdYlGn")

    st.dataframe(sty, width="stretch", height=420)

    # ---------------------------------------------------------------------
    # Inspect + Replay
    # ---------------------------------------------------------------------
    st.write("#### Inspect a strategy")
    default_pick = st.session_state.get("ui.batch.inspect_id")
    if default_pick not in set(df_show["config_id"].astype(str)):
        default_pick = str(df_show["config_id"].iloc[0]) if not df_show.empty else None

    if default_pick:
        pick = st.selectbox(
            "Choose a config_id to inspect / replay",
            options=df_show["config_id"].astype(str).tolist(),
            index=df_show["config_id"].astype(str).tolist().index(default_pick),
            key="ui.batch.pick",
        )
        batch_profit_ex_cf = None
        row = df_show[df_show["config_id"].astype(str) == str(pick)].head(1)
        if not row.empty:
            r = row.iloc[0].to_dict()
            # Keep a reference for replay sanity checks
            batch_profit_ex_cf = None
            try:
                if profit_col:
                    batch_profit_ex_cf = float(r.get(profit_col))
            except Exception:
                batch_profit_ex_cf = None

            c1, c2, c3, c4 = st.columns(4)
            if profit_col:
                c1.metric("Profit (ex cashflows)", _fmt_money(r.get(profit_col)))
            if dd_col:
                c2.metric("Max DD", _fmt_pct(_drawdown_to_frac(pd.Series([r.get(dd_col)])).iloc[0]))
            if trades_col:
                c3.metric("Trades", f"{int(float(r.get(trades_col, 0) or 0))}")
            if calmar_col:
                c4.metric("Calmar", _fmt_num(r.get(calmar_col), digits=3))

        # Artifacts locations
        replay_dir = run_dir / "replay_cache" / str(pick)
        art_dir = replay_dir if (replay_dir / "equity_curve.csv").exists() else top_map.get(str(pick), replay_dir)

        replay_dl_items: List[Tuple[str, bytes, str]] = []
        eq_path = art_dir / "equity_curve.csv"
        can_replay = (run_dir / "configs_resolved.jsonl").exists()
        if not eq_path.exists():
            if st.button("Generate replay artifacts (cached)", type="primary", disabled=(not can_replay), key="batch.replay.btn"):
                progress_path = replay_dir / "progress.jsonl"
                meta = {}
                try:
                    mp = run_dir / "batch_meta.json"
                    if mp.exists():
                        meta = _read_json(mp)
                except Exception:
                    meta = {}
                starting_equity = float(meta.get("starting_equity", 10000.0) or 10000.0)
                seed = int(meta.get("seed", 1) or 1)

                cmd = [
                    PY,
                    str(REPO_ROOT / "tools" / "generate_replay_artifacts.py"),
                    "--from-run",
                    str(run_dir),
                    "--config-id",
                    str(pick),
                    "--progress-file",
                    str(progress_path),
                    "--starting-equity",
                    str(starting_equity),
                    "--seed",
                    str(seed),
                ]
                _run_cmd(cmd, cwd=REPO_ROOT, label="Replay: generate artifacts", progress_path=progress_path)
                st.success("Replay artifacts generated.")
                st.rerun()

        if eq_path.exists():
            try:
                eq_df = pd.read_csv(eq_path)
                st.caption(f"Replay artifacts dir: `{art_dir}`")

                if "equity" in eq_df.columns:
                    # Use dt column when present, otherwise fall back to first column
                    tcol = "dt" if "dt" in eq_df.columns else eq_df.columns[0]
                    t = pd.to_datetime(eq_df[tcol], errors="coerce")
                    equity = pd.to_numeric(eq_df["equity"], errors="coerce")

                    start_eq = float(equity.iloc[0]) if len(equity) else float("nan")
                    end_eq = float(equity.iloc[-1]) if len(equity) else float("nan")
                    cashflow_total = 0.0
                    if "cashflow" in eq_df.columns:
                        cashflow_total = float(pd.to_numeric(eq_df["cashflow"], errors="coerce").fillna(0.0).sum())
                    replay_profit_ex_cf = float(end_eq - start_eq - cashflow_total) if (math.isfinite(end_eq) and math.isfinite(start_eq)) else float("nan")

                    cA, cB, cC, cD = st.columns(4)
                    cA.metric("Replay start equity", _fmt_money(start_eq))
                    cB.metric("Replay end equity", _fmt_money(end_eq))
                    cC.metric("Replay cashflows", _fmt_money(cashflow_total))
                    cD.metric("Replay profit (ex cashflows)", _fmt_money(replay_profit_ex_cf))

                    # Warn when replay doesn't match the batch row (common sign of loading the wrong config/dataset)
                    try:
                        if batch_profit_ex_cf is not None and math.isfinite(replay_profit_ex_cf):
                            if abs(float(batch_profit_ex_cf) - float(replay_profit_ex_cf)) > max(5.0, 0.01 * abs(float(batch_profit_ex_cf))):
                                st.warning(
                                    "Replay result does **not** match the batch row profit. "
                                    "This usually means the replay loaded the wrong config payload (e.g., didn't unwrap normalized config), "
                                    "or the replay is using a different dataset/seed/starting equity."
                                )
                    except Exception:
                        pass

                                        # -----------------------------------------------------------------
                    # Replay charts (juicy + interpretable)
                    # -----------------------------------------------------------------
                    trades_path = art_dir / "trades.csv"
                    fills_path = art_dir / "fills.csv"

                    # Build a plot-friendly frame
                    try:
                        t_idx = t if t.notna().any() else pd.to_datetime(eq_df[tcol], errors="coerce")
                    except Exception:
                        t_idx = pd.to_datetime(eq_df[tcol], errors="coerce")

                    plot_df = pd.DataFrame(
                        {
                            "dt": t_idx,
                            "equity": pd.to_numeric(eq_df.get("equity"), errors="coerce"),
                            "cash": pd.to_numeric(eq_df.get("cash"), errors="coerce") if "cash" in eq_df.columns else np.nan,
                            "price": pd.to_numeric(eq_df.get("price"), errors="coerce") if "price" in eq_df.columns else np.nan,
                            "pos_qty": pd.to_numeric(eq_df.get("pos_qty"), errors="coerce") if "pos_qty" in eq_df.columns else np.nan,
                            "cashflow": pd.to_numeric(eq_df.get("cashflow"), errors="coerce") if "cashflow" in eq_df.columns else 0.0,
                        }
                    )
                    plot_df = plot_df.dropna(subset=["dt"]).sort_values("dt")
                    plot_df["pos_value"] = plot_df["pos_qty"] * plot_df["price"] if ("pos_qty" in plot_df.columns and "price" in plot_df.columns) else np.nan
                    plot_df["exposure"] = np.nan
                    try:
                        pv = pd.to_numeric(plot_df["pos_value"], errors="coerce")
                        eqv = pd.to_numeric(plot_df["equity"], errors="coerce")
                        plot_df["exposure"] = (pv.abs() / eqv.replace(0.0, np.nan)).clip(0.0, 5.0)
                    except Exception:
                        pass

                    # Load trades for markers (if present)
                    td = None
                    if trades_path.exists():
                        try:
                            td = pd.read_csv(trades_path)
                        except Exception:
                            td = None

                    def _nearest_y(ts: pd.Series) -> np.ndarray:
                        # Nearest equity values for a series of datetimes
                        x = plot_df["dt"].to_numpy(dtype="datetime64[ns]")
                        y = plot_df["equity"].to_numpy(dtype=float)
                        t_arr = pd.to_datetime(ts, errors="coerce").to_numpy(dtype="datetime64[ns]")
                        out = np.full(len(t_arr), np.nan, dtype=float)
                        if len(x) == 0:
                            return out
                        # Searchsorted gives insertion point; compare neighbors to pick nearest
                        idxs = np.searchsorted(x, t_arr, side="left")
                        idxs = np.clip(idxs, 0, len(x) - 1)
                        prev = np.clip(idxs - 1, 0, len(x) - 1)
                        # Choose prev when it's closer
                        choose_prev = np.abs(t_arr - x[prev]) <= np.abs(x[idxs] - t_arr)
                        best = np.where(choose_prev, prev, idxs)
                        out = y[best]
                        return out

                    tab_eq, tab_cash, tab_exp = st.tabs(["Equity + Trades", "Cash vs Position", "Exposure"])

                    with tab_eq:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=plot_df["dt"], y=plot_df["equity"], mode="lines", name="Equity"))

                        # Deposit / cashflow markers (spot/DCA interpretable moment)
                        try:
                            cf = pd.to_numeric(plot_df["cashflow"], errors="coerce").fillna(0.0)
                            mask = cf.abs() > 1e-9
                            if mask.any():
                                fig.add_trace(
                                    go.Scatter(
                                        x=plot_df.loc[mask, "dt"],
                                        y=plot_df.loc[mask, "equity"],
                                        mode="markers",
                                        name="Cashflow",
                                        marker=dict(size=6, symbol="circle-open"),
                                        hovertemplate="dt=%{x}<br>cashflow=%{customdata:,.2f}<extra></extra>",
                                        customdata=cf.loc[mask].to_numpy(),
                                    )
                                )
                        except Exception:
                            pass

                        # Trade markers
                        if td is not None and not td.empty and "entry_dt" in td.columns:
                            try:
                                entry_t = pd.to_datetime(td["entry_dt"], errors="coerce")
                                entry_y = _nearest_y(entry_t)
                                fig.add_trace(
                                    go.Scatter(
                                        x=entry_t,
                                        y=entry_y,
                                        mode="markers",
                                        name="Entry",
                                        marker=dict(size=8, symbol="triangle-up"),
                                        hovertemplate="entry=%{x}<extra></extra>",
                                    )
                                )
                            except Exception:
                                pass

                        if td is not None and not td.empty and "exit_dt" in td.columns:
                            try:
                                exit_t = pd.to_datetime(td["exit_dt"], errors="coerce")
                                exit_mask = exit_t.notna()
                                exit_t2 = exit_t[exit_mask]
                                exit_y = _nearest_y(exit_t2)
                                # If we have net_pnl, pass it as customdata so you can see winners/losers on hover
                                custom = None
                                if "net_pnl" in td.columns:
                                    custom = pd.to_numeric(td.loc[exit_mask, "net_pnl"], errors="coerce").to_numpy()
                                fig.add_trace(
                                    go.Scatter(
                                        x=exit_t2,
                                        y=exit_y,
                                        mode="markers",
                                        name="Exit",
                                        marker=dict(size=8, symbol="triangle-down"),
                                        hovertemplate="exit=%{x}<br>net_pnl=%{customdata:,.2f}<extra></extra>",
                                        customdata=custom,
                                    )
                                )
                            except Exception:
                                pass

                        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=420, legend=dict(orientation="h"))
                        _plotly(fig)

                        # Drawdown (with max-DD episode shading)
                        try:
                            eq2 = pd.to_numeric(plot_df["equity"], errors="coerce").fillna(method="ffill")
                            dt2 = pd.to_datetime(plot_df["dt"], errors="coerce")
                            peak = eq2.cummax()
                            dd = (eq2 / peak - 1.0).fillna(0.0)

                            # Identify max drawdown episode (peak -> trough -> recovery if any)
                            x0 = None
                            x1 = None
                            try:
                                if len(dd):
                                    trough_i = int(np.nanargmin(dd.to_numpy()))
                                    peak_val = float(peak.iloc[trough_i])

                                    # start = last time equity touched the high-water mark before the trough
                                    pre_eq = eq2.iloc[:trough_i]  # exclude trough itself
                                    if len(pre_eq):
                                        hit = pre_eq >= peak_val * (1.0 - 1e-9)
                                        if hit.any():
                                            start_pos = int(np.flatnonzero(hit.to_numpy())[-1])
                                            x0 = dt2.iloc[start_pos]

                                    # recovery = first time AFTER the trough equity >= peak
                                    post_eq = eq2.iloc[trough_i + 1 :]
                                    if len(post_eq):
                                        rec_mask = post_eq >= peak_val * (1.0 - 1e-9)
                                        if rec_mask.any():
                                            rec_rel = int(np.flatnonzero(rec_mask.to_numpy())[0])
                                            x1 = dt2.iloc[trough_i + 1 + rec_rel]

                                    # If never recovered, shade peak->trough so the user still sees "the bad zone"
                                    if x0 is not None and x1 is None:
                                        x1 = dt2.iloc[min(trough_i, len(dt2) - 1)]
                            except Exception:
                                x0 = None
                                x1 = None

                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatter(x=dt2, y=dd, mode="lines", name="Drawdown"))

                            if x0 is not None and x1 is not None and pd.notna(x0) and pd.notna(x1):
                                fig2.add_vrect(
                                    x0=x0,
                                    x1=x1,
                                    fillcolor="rgba(200,0,0,0.08)",
                                    line_width=0,
                                    annotation_text="Max DD episode",
                                    annotation_position="top left",
                                    annotation_font_size=10,
                                )

                            fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=260, legend=dict(orientation="h"))
                            _plotly(fig2)
                        except Exception:
                            pass

                    with tab_cash:
                        fig = go.Figure()
                        if "cash" in plot_df.columns:
                            fig.add_trace(go.Scatter(x=plot_df["dt"], y=plot_df["cash"], mode="lines", name="Cash"))
                        if "pos_value" in plot_df.columns:
                            fig.add_trace(go.Scatter(x=plot_df["dt"], y=plot_df["pos_value"], mode="lines", name="Position value"))
                        fig.add_trace(go.Scatter(x=plot_df["dt"], y=plot_df["equity"], mode="lines", name="Equity"))
                        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360, legend=dict(orientation="h"))
                        _plotly(fig)

                    with tab_exp:
                        if "exposure" in plot_df.columns and plot_df["exposure"].notna().any():
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=plot_df["dt"], y=plot_df["exposure"], mode="lines", name="Exposure (pos_value / equity)"))
                            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320, legend=dict(orientation="h"))
                            _plotly(fig)
                        else:
                            st.caption("Exposure chart unavailable (missing pos_qty/price/equity columns).")


                    # -----------------------------------------------------------------
                    # Strategy story (interpretable summary from replay artifacts)
                    # -----------------------------------------------------------------
                    with st.expander("Strategy story (replay)", expanded=True):
                        # --- Drawdown story ---
                        try:
                            eq_series = pd.to_numeric(plot_df["equity"], errors="coerce").fillna(method="ffill")
                            dt_series = pd.to_datetime(plot_df["dt"], errors="coerce")
                            peak = eq_series.cummax()
                            dd = (eq_series / peak - 1.0).fillna(0.0)

                            dd_min = float(dd.min()) if len(dd) else float("nan")
                            dd_trough_i = int(np.nanargmin(dd.to_numpy())) if len(dd) else None

                            dd_start_dt = None
                            dd_trough_dt = None
                            dd_recover_dt = None
                            dd_peak_val = None

                            if dd_trough_i is not None and math.isfinite(dd_min):
                                dd_trough_dt = dt_series.iloc[dd_trough_i]
                                dd_peak_val = float(peak.iloc[dd_trough_i])
                            
                                # peak date = last time equity touched the high-water mark before the trough
                                pre_eq = eq_series.iloc[:dd_trough_i]  # exclude trough itself
                                if len(pre_eq):
                                    hit = pre_eq >= dd_peak_val * (1.0 - 1e-9)
                                    if hit.any():
                                        start_pos = int(np.flatnonzero(hit.to_numpy())[-1])
                                        dd_start_dt = dt_series.iloc[start_pos]
                                    else:
                                        dd_start_dt = dt_series.iloc[0]
                                else:
                                    dd_start_dt = dt_series.iloc[0]
                            
                                # recovery date = first time AFTER trough equity >= peak value
                                post_eq = eq_series.iloc[dd_trough_i + 1 :]
                                if len(post_eq):
                                    rec_mask = post_eq >= dd_peak_val * (1.0 - 1e-9)
                                    if rec_mask.any():
                                        rec_rel = int(np.flatnonzero(rec_mask.to_numpy())[0])
                                        dd_recover_dt = dt_series.iloc[dd_trough_i + 1 + rec_rel]

                            # underwater longest segment
                            underwater = eq_series < peak * (1.0 - 1e-12)
                            uw = underwater.astype(int)
                            seg = (uw.diff().fillna(0).abs() > 0).cumsum()

                            longest_days = None
                            longest_start = None
                            longest_end = None
                            if underwater.any():
                                for sid, grp in plot_df.loc[underwater].groupby(seg[underwater]):
                                    dts = pd.to_datetime(grp["dt"], errors="coerce")
                                    if dts.notna().any():
                                        dur = (dts.max() - dts.min()).total_seconds() / 86400.0
                                        if (longest_days is None) or (dur > longest_days):
                                            longest_days = dur
                                            longest_start = dts.min()
                                            longest_end = dts.max()

                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Max drawdown", _fmt_pct(dd_min))
                            c2.metric("DD peak", f"{dd_start_dt.date()}" if dd_start_dt is not None else "n/a")
                            c3.metric("DD trough", f"{dd_trough_dt.date()}" if dd_trough_dt is not None else "n/a")
                            if dd_recover_dt is not None and dd_start_dt is not None:
                                days = (dd_recover_dt - dd_start_dt).total_seconds() / 86400.0
                                c4.metric("DD to recovery", f"{days:.0f} days")
                                st.caption(f"Recovery: {dd_recover_dt.date()}")
                            else:
                                c4.metric("DD to recovery", "not recovered" if dd_start_dt is not None else "n/a")

                            if longest_days is not None and longest_start is not None and longest_end is not None:
                                st.caption(f"Longest underwater: {longest_days:.0f} days ({longest_start.date()} → {longest_end.date()})")
                        except Exception:
                            st.info("Drawdown story unavailable (missing equity series).")

                        st.divider()

                        # --- Trades story ---
                        if td is None or td.empty:
                            st.info("No trades.csv found for this replay yet.")
                        else:
                            tdf = td.copy()
                            if "net_pnl" in tdf.columns:
                                pnl = pd.to_numeric(tdf["net_pnl"], errors="coerce")
                            elif "pnl" in tdf.columns:
                                pnl = pd.to_numeric(tdf["pnl"], errors="coerce")
                            else:
                                pnl = pd.Series([np.nan] * len(tdf))

                            wins = pnl > 0
                            win_rate = float(wins.mean()) if len(pnl.dropna()) else float("nan")
                            gross_win = float(pnl[wins].sum()) if wins.any() else 0.0
                            gross_loss = float(pnl[~wins].sum()) if (~wins).any() else 0.0
                            profit_factor = (gross_win / abs(gross_loss)) if gross_loss < 0 else float("inf") if gross_win > 0 else float("nan")

                            best_trade = float(pnl.max()) if pnl.notna().any() else float("nan")
                            worst_trade = float(pnl.min()) if pnl.notna().any() else float("nan")

                            hold_days = None
                            if "entry_dt" in tdf.columns and "exit_dt" in tdf.columns:
                                ed = pd.to_datetime(tdf["entry_dt"], errors="coerce")
                                xd = pd.to_datetime(tdf["exit_dt"], errors="coerce")
                                hold = (xd - ed).dt.total_seconds() / 86400.0
                                hold_days = float(hold.mean()) if hold.notna().any() else None

                            exp_mean = float(pd.to_numeric(plot_df.get("exposure"), errors="coerce").mean()) if "exposure" in plot_df.columns else float("nan")
                            exp_max = float(pd.to_numeric(plot_df.get("exposure"), errors="coerce").max()) if "exposure" in plot_df.columns else float("nan")

                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Win rate", _fmt_pct(win_rate))
                            c2.metric("Profit factor", _fmt_num(profit_factor, digits=2) if math.isfinite(profit_factor) else "∞" if profit_factor == float("inf") else "n/a")
                            c3.metric("Best / Worst trade", f"{_fmt_money(best_trade)} / {_fmt_money(worst_trade)}")
                            c4.metric("Avg exposure", _fmt_pct(exp_mean))

                            if hold_days is not None:
                                st.caption(f"Avg hold time: {hold_days:.1f} days · Max exposure: {_fmt_pct(exp_max)}")

                            show_cols = []
                            for c in ["entry_dt", "exit_dt", "entry_price", "exit_price", "qty", "net_pnl", "fees", "reason", "exit_reason"]:
                                if c in tdf.columns:
                                    show_cols.append(c)
                            if "net_pnl" not in tdf.columns:
                                tdf["net_pnl"] = pnl

                            st.write("**Top trades**")
                            left, right = st.columns(2)
                            with left:
                                st.caption("Best (by net_pnl)")
                                st.dataframe(tdf.sort_values("net_pnl", ascending=False).head(5)[show_cols or ["net_pnl"]], width="stretch", height=220)
                            with right:
                                st.caption("Worst (by net_pnl)")
                                st.dataframe(tdf.sort_values("net_pnl", ascending=True).head(5)[show_cols or ["net_pnl"]], width="stretch", height=220)

                        st.divider()

                        # --- Cashflow story ---
                        try:
                            cf = pd.to_numeric(plot_df.get("cashflow"), errors="coerce").fillna(0.0)
                            mask = cf.abs() > 1e-9
                            if mask.any():
                                n = int(mask.sum())
                                total = float(cf[mask].sum())
                                avg = float(cf[mask].mean())
                                st.write("**Cashflows (deposits/withdrawals)**")
                                c1, c2, c3 = st.columns(3)
                                c1.metric("Cashflow events", f"{n}")
                                c2.metric("Total cashflow", _fmt_money(total))
                                c3.metric("Avg event size", _fmt_money(avg))
                            else:
                                st.caption("No cashflows recorded in equity curve.")
                        except Exception:
                            pass

                # Stash replay exports for the Exports dropdown below
                replay_dl_items = [
                    (
                        "Download replay equity_curve.csv",
                        eq_path.read_bytes(),
                        f"{selected_run_name}_{pick}_equity_curve.csv",
                    )
                ]
                trades_path = art_dir / "trades.csv"
                fills_path = art_dir / "fills.csv"
                if trades_path.exists():
                    replay_dl_items.append(
                        (
                            "Download replay trades.csv",
                            trades_path.read_bytes(),
                            f"{selected_run_name}_{pick}_trades.csv",
                        )
                    )
                if fills_path.exists():
                    replay_dl_items.append(
                        (
                            "Download replay fills.csv",
                            fills_path.read_bytes(),
                            f"{selected_run_name}_{pick}_fills.csv",
                        )
                    )
            except Exception as e:
                st.warning(f"Could not load replay artifacts: {e}")
        else:
            st.caption("Replay artifacts not found yet for this config_id.")
    # ---------------------------------------------------------------------
    # Exports (advanced)
    # ---------------------------------------------------------------------
    with st.expander("Exports (advanced)", expanded=False):
        if replay_dl_items:
            st.write("**Replay exports**")
            for label, data, fname in replay_dl_items:
                st.download_button(label, data=data, file_name=fname)
            st.divider()

        st.write("**Batch exports**")
        st.download_button(
            "Download batch survivors (CSV)",
            data=df_show.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_run_name}_batch_view.csv",
        )

        show_raw = st.checkbox("Show raw table", value=False, key="ui.batch.show_raw")
        if show_raw:
                cols = [
                    "config_id",
                    "config.label",
                    "batch.verdict",
                    "equity.net_profit_ex_cashflows",
                    "performance.twr_total_return",
                    "performance.max_drawdown_equity",
                    "trades_summary.trades_closed",
                ]
                for c in ["score.calmar_equity", "score.profit_dd", "score.twr_dd", "score.profit"]:
                    if c in df_show.columns and c not in cols:
                        cols.append(c)
                cols = [c for c in cols if c in df_show.columns]
                st.dataframe(df_show[cols], width="stretch", height=520)

    st.divider()
    c_next, _ = st.columns([1, 4])
    with c_next:
        if st.button("Next: Rolling Starts →", type="primary"):
            st.session_state["ui.stage"] = "rs"
            st.rerun()
    st.caption("Next: run Rolling Starts to measure start-date fragility.")

# =============================================================================
# Stage B: Rolling Starts
# =============================================================================

if stage_pick == "rs":
    st.write("### B) Rolling Starts (start-date sensitivity)")
    with st.expander("How to read Rolling Starts", expanded=True):
        st.markdown(
            """
Rolling Starts reruns the **same** strategy many times — each run starts on a different date.
It answers the quant-flavored question: **is this edge real, or is it just “you started on the right day”?**

**How to read the summary numbers**
- **Windows**: number of start dates tested. More is better. **< 10 is noisy**.
- **Return p10 / p50 / p90**: pessimistic / typical / optimistic outcomes across start dates.
- **DD p90**: a “bad-but-plausible” max drawdown. Lower is better.
- **Underwater p90 (days)**: how long you’re likely to be stuck below your prior equity peak.
- **Fragility (spread p90 − p10)**: how wide outcomes swing across start dates. **Smaller = more stable.**

**How to read the charts**
- Each dot = one rolling-start window (one start date).
- Tight clusters are good. Wild scatter means **start-date luck** dominates.
- Dashed line = median (p50). Dotted lines = p10 / p90.

Rule of thumb: prefer strategies with a **decent p10** (survives bad starts), not just a spicy p50.
            """
        )

    st.caption("Runs the same strategy many times with different starting days, to measure fragility.")

    # RS selection / settings
    left, right = st.columns([2, 1])

    with left:
        rs_runs = []
        if rs_root.exists():
            rs_runs = [p for p in rs_root.glob("rs_*") if p.is_dir()]
            rs_runs = sorted(rs_runs, key=lambda p: p.stat().st_mtime, reverse=True)

        rs_choice = st.selectbox(
            "Rolling-start runs found",
            options=["(none)"] + [p.name for p in rs_runs],
            index=(1 if rs_runs else 0),
            key="rs.pick",
        )
        rs_dir = (rs_root / rs_choice) if (rs_choice != "(none)") else None

        with right:
            st.write("**Quick presets**")

            bars_per_day = _bars_per_day_from_run_meta(run_dir)
            bar_hint = _human_bar_interval_from_run(run_dir)
            st.caption(f"Detected timeframe: {bar_hint} (≈ {bars_per_day} bars/day)")

            preset = st.selectbox("Preset", options=["Quick", "Standard", "Thorough"], index=0, key="rs.preset")

            # Apply defaults only when preset changes (so number inputs don't reset constantly).
            prev = st.session_state.get("rs.preset_prev")
            if prev != preset:
                if bars_per_day <= 2:
                    # Daily-ish bars: space starts out in days, and require a long minimum history
                    if preset == "Quick":
                        step_days, min_days = 30, 365
                    elif preset == "Standard":
                        step_days, min_days = 14, 365
                    else:
                        step_days, min_days = 7, 365
                else:
                    # Intraday: still think in calendar days (convert to bars), but min history can be shorter
                    if preset == "Quick":
                        step_days, min_days = 7, 60
                    elif preset == "Standard":
                        step_days, min_days = 3, 90
                    else:
                        step_days, min_days = 1, 120

                st.session_state["rs.start_step"] = int(max(1, round(step_days * bars_per_day)))
                st.session_state["rs.min_bars"] = int(max(30, round(min_days * bars_per_day)))
                st.session_state["rs.preset_prev"] = preset

            start_step = int(
                st.number_input(
                    "Start step (bars)",
                    1,
                    500000,
                    int(st.session_state.get("rs.start_step", max(1, int(round(7 * bars_per_day))))),
                    5,
                    key="rs.start_step",
                )
            )
            min_bars = int(
                st.number_input(
                    "Min bars per start",
                    30,
                    5000000,
                    int(st.session_state.get("rs.min_bars", max(30, int(round(60 * bars_per_day))))),
                    30,
                    key="rs.min_bars",
                )
            )
            st.caption(f"Preset interpretation: start every ~{max(1, round(start_step / max(1e-9, bars_per_day))):.0f} days; require ~{max(1, round(min_bars / max(1e-9, bars_per_day))):.0f} days of data per start.")

            min_bars_effective = int(min_bars)  # placeholder for future clamping logic

    # Compute survivor ids (we stress test every survivor in full_passed if available, else sweep_passed)
    survivors_ids = survivors["config_id"].astype(str).tolist()
    ids_file = run_dir / "post" / "survivor_ids.txt"
    ids_file.parent.mkdir(parents=True, exist_ok=True)
    ids_file.write_text("\n".join(survivors_ids) + "\n", encoding="utf-8")

    rs_out_dir = rs_root / f"rs_step{start_step}_min{min_bars}_n{len(survivors_ids)}"
    st.caption(f"Will run on survivors: {len(survivors_ids)} configs → output: {rs_out_dir}")

    can_run = True
    if len(survivors_ids) == 0:
        can_run = False
        st.error("No survivors IDs found.")
    if not (run_dir / "configs_resolved.jsonl").exists():
        can_run = False
        st.error("Missing configs_resolved.jsonl (needed for rolling starts).")

    if st.button("Run Rolling Starts for all survivors", type="primary", disabled=(not can_run)):
        try:
            cmd = [
                PY,
                "-m",
                "research.rolling_starts",
                "--from-run",
                str(run_dir),
                "--out",
                str(rs_out_dir),                "--top-n",
                str(len(survivors_ids)),
                "--start-step",
                str(start_step),
                "--min-bars",
                str(min_bars_effective),
                "--seed",
                "1",
                "--starting-equity",
                str(float(meta.get("starting_equity", 1000.0) or 1000.0)),
                "--jobs", "8",
            ]
            rs_progress = rs_out_dir / "progress" / "rolling_starts.jsonl"
            rs_progress.parent.mkdir(parents=True, exist_ok=True)
            cmd += ["--no-progress", "--progress-file", str(rs_progress), "--progress-every", "25"]
            _run_cmd(cmd, cwd=REPO_ROOT, label="Rolling Starts", progress_path=rs_progress)
            st.success("Rolling Starts complete.")
            st.rerun()
        except Exception as e:
            st.error(str(e))
            st.stop()

    # Load chosen/latest summary
    rs_dir_effective = rs_dir or rs_out_dir
    rs_sum = load_rs_summary(run_dir, rs_dir_effective)
    rs_det = load_rs_detail(run_dir, rs_dir_effective)

    if rs_sum is None or rs_sum.empty:
        st.info("No rolling-start stats found yet for the chosen output folder.")
        st.stop()

    # Merge + evaluate
    base = survivors.copy()
    base = merge_stage(base, rs_sum, on="config_id", suffix="rs")

    cov = int(base["rs.measured"].sum()) if "rs.measured" in base.columns else 0
    st.success(f"Coverage: {cov}/{len(base)} configs have rolling-start stats in this folder.")

    with st.expander("Rolling-start questions (filters)", expanded=True):
        rs_ans = _question_ui(rolling_questions(), key_prefix="q.rs")

    dfB = apply_stage_eval(base, stage_key="rsq", questions=rolling_questions(), answers=rs_ans)

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        show_pass = st.checkbox("Show PASS", value=True, key="f.rs.pass")
    with col2:
        show_warn = st.checkbox("Show WARN", value=True, key="f.rs.warn")
    with col3:
        show_fail = st.checkbox("Show FAIL", value=False, key="f.rs.fail")

    keep = []
    if show_pass:
        keep.append("PASS")
    if show_warn:
        keep.append("WARN")
    if show_fail:
        keep.append("FAIL")
    df_show = dfB[dfB["rsq.verdict"].isin(keep)].copy()

    
if stage_pick == "rs":
    # -------------------------------------------------------------------------
    # Rolling Starts: interpretability
    # -------------------------------------------------------------------------
    # Main table columns
    cols = [
        "config_id",
        "config.label",
        "rsq.verdict",
        "twr_p10",
        "twr_p50",
        "twr_p90",
        "dd_p50",
        "dd_p90",
        "uw_p50_days",
        "uw_p90_days",
        "util_p50",
        "util_p90",
        "robustness_score",
        "windows",
    ]
    cols = [c for c in cols if c in df_show.columns]
    if "rs.measured" in df_show.columns:
        cols.insert(2, "rs.measured")

    # Quick visual scan: robustness map (return vs drawdown)
    st.subheader("Quick visual scan")
    df_plot = df_show.copy()
    for c in ["twr_p10", "twr_p50", "twr_p90", "dd_p50", "dd_p90", "robustness_score", "windows"]:
        if c in df_plot.columns:
            df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")


    # Derived measures (UI-only)
    if ("twr_p10" in df_plot.columns) and ("twr_p90" in df_plot.columns):
        df_plot["fragility_spread"] = pd.to_numeric(df_plot["twr_p90"], errors="coerce") - pd.to_numeric(df_plot["twr_p10"], errors="coerce")
    else:
        df_plot["fragility_spread"] = np.nan

    # (Micro-juice) Let users tighten the frontier so it doesn't get dominated by tiny sample sizes.
    min_windows = int(st.slider("Minimum rolling-start windows", 1, 200, 10, 1, key="rs.min_windows_ui"))
    # Filter by minimum windows (avoid df.get(...) returning an int when the column is missing)
    if "windows" in df_plot.columns:
        _win = pd.to_numeric(df_plot["windows"], errors="coerce").fillna(0).astype(int)
        df_plot = df_plot[_win >= min_windows].copy()
    else:
        st.info("No rolling-start 'windows' column found yet (run Rolling Starts first).")
        df_plot = df_plot.iloc[0:0].copy()



    # Cohort summary + fragility labels (relative to the current table)
    if not df_plot.empty:
        _sp = pd.to_numeric(df_plot.get("fragility_spread"), errors="coerce")
        if _sp.notna().any():
            q33 = float(_sp.quantile(0.33))
            q66 = float(_sp.quantile(0.66))
        else:
            q33, q66 = float("nan"), float("nan")

        def _frag_label(v: Any) -> str:
            try:
                x = float(v)
                if not math.isfinite(x):
                    return "—"
                if math.isfinite(q33) and x <= q33:
                    return "Low"
                if math.isfinite(q66) and x <= q66:
                    return "Medium"
                return "High"
            except Exception:
                return "—"

        df_plot["fragility"] = df_plot.get("fragility_spread").apply(_frag_label) if "fragility_spread" in df_plot.columns else "—"

        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Strategies (≥ min windows)", f"{len(df_plot)}")
        if "twr_p10" in df_plot.columns:
            cc2.metric("Median return p10", _fmt_pct(pd.to_numeric(df_plot["twr_p10"], errors="coerce").median()))
        if "dd_p90" in df_plot.columns:
            cc3.metric("Median DD p90", _fmt_pct(pd.to_numeric(df_plot["dd_p90"], errors="coerce").median()))
        if "fragility" in df_plot.columns:
            high_share = (df_plot["fragility"] == "High").mean() if len(df_plot) else 0.0
            cc4.metric("High fragility share", f"{high_share*100:.0f}%")

        st.caption("Each dot = one strategy. Up/right is better (higher median return, lower DD p90). Bigger dots = more rolling-start windows.")

        color_mode = st.selectbox("Color dots by", ["Verdict", "Fragility"], index=0, key="rs.scan.color")
        if color_mode == "Fragility" and "fragility" in df_plot.columns:
            color_col = "fragility"
        else:
            color_col = ("rsq.verdict" if "rsq.verdict" in df_plot.columns else None)
    else:
        color_col = None
    if df_plot.empty or ("dd_p90" not in df_plot.columns) or ("twr_p50" not in df_plot.columns):
        st.info("Not enough Rolling Starts data to plot yet.")
    else:
        fig = px.scatter(
            df_plot,
            x="dd_p90",
            y="twr_p50",
            color=color_col,
            size=("windows" if "windows" in df_plot.columns else None),
            hover_data=[c for c in ["config_id", "config.label", "rsq.verdict", "twr_p10", "twr_p50", "twr_p90", "dd_p90", "uw_p90_days", "windows", "fragility", "fragility_spread", "robustness_score"] if c in df_plot.columns],
            labels={
                "dd_p90": "Drawdown p90 (lower is better)",
                "twr_p50": "Return p50 (higher is better)",
            },
            title="Rolling Starts: return vs drawdown (robustness map)",
        )
        _plotly(fig)

    st.subheader("Inspect a strategy (Rolling Starts)")
    if df_show.empty:
        st.info("No Rolling Starts rows to inspect.")
    else:
        # Pick a config to inspect (default: best robustness_score)
        df_rank = df_show.copy()
        if "robustness_score" in df_rank.columns:
            df_rank["robustness_score"] = pd.to_numeric(df_rank["robustness_score"], errors="coerce")
            df_rank = df_rank.sort_values("robustness_score", ascending=False)
        inspect_opts = df_rank["config_id"].astype(str).tolist()
        inspect_id = st.selectbox("Choose a config_id", options=inspect_opts, index=0, key="rs.inspect_id")

        row = df_show[df_show["config_id"].astype(str) == str(inspect_id)].head(1)
        if row.empty:
            st.warning("Could not load that config_id from the Rolling Starts view.")
        else:
            r0 = row.iloc[0]

            def _fmt_pct(x: float) -> str:
                try:
                    if pd.isna(x):
                        return "—"
                    return f"{float(x)*100:.2f}%"
                except Exception:
                    return "—"

            def _fmt_num(x: float) -> str:
                try:
                    if pd.isna(x):
                        return "—"
                    return f"{float(x):.3f}"
                except Exception:
                    return "—"

            # Mini story cards
            c1, c2, c3, c4, c5 = st.columns(5)
            if "windows" in row.columns:
                c1.metric("Windows", f"{int(pd.to_numeric(r0.get('windows'), errors='coerce') or 0)}")
            else:
                c1.metric("Windows", "—")
            c2.metric("Return p10", _fmt_pct(r0.get("twr_p10")))
            c3.metric("Return p50", _fmt_pct(r0.get("twr_p50")))
            c4.metric("DD p90", _fmt_pct(r0.get("dd_p90")))
            if "robustness_score" in row.columns:
                c5.metric("Robustness score", _fmt_num(r0.get("robustness_score")))
            else:
                c5.metric("Robustness score", "—")

            # Fragility: a fast, plain-English read (start-date sensitivity)
            frag = None
            try:
                frag = float(r0.get("twr_p90")) - float(r0.get("twr_p10"))
            except Exception:
                frag = None

            # Label fragility relative to the current cohort (so it adapts to different datasets)
            label = "—"
            try:
                cohort = df_show.copy()
                if ("twr_p10" in cohort.columns) and ("twr_p90" in cohort.columns):
                    cohort["_spread"] = pd.to_numeric(cohort["twr_p90"], errors="coerce") - pd.to_numeric(cohort["twr_p10"], errors="coerce")
                    cohort["_spread"] = cohort["_spread"].replace([np.inf, -np.inf], np.nan)
                    cohort_sp = cohort["_spread"].dropna()
                    if len(cohort_sp) >= 8 and frag is not None and math.isfinite(float(frag)):
                        q33 = float(cohort_sp.quantile(0.33))
                        q66 = float(cohort_sp.quantile(0.66))
                        if float(frag) <= q33:
                            label = "Low"
                        elif float(frag) <= q66:
                            label = "Medium"
                        else:
                            label = "High"
            except Exception:
                label = "—"

            # Plain-English reason (what hurts, how often)
            uw_p90 = r0.get("uw_p90_days") if "uw_p90_days" in row.columns else None
            win_n = None
            try:
                win_n = int(pd.to_numeric(r0.get("windows"), errors="coerce") or 0)
            except Exception:
                win_n = None

            parts = []
            if frag is not None and not pd.isna(frag):
                parts.append(f"spread(p90−p10) {_fmt_pct(frag)}")
            if "twr_p10" in row.columns:
                parts.append(f"p10 {_fmt_pct(r0.get('twr_p10'))}")
            if "dd_p90" in row.columns:
                parts.append(f"DD p90 {_fmt_pct(r0.get('dd_p90'))}")
            if uw_p90 is not None and not pd.isna(uw_p90):
                parts.append(f"UW p90 {int(float(uw_p90))}d")
            if win_n is not None:
                parts.append(f"{win_n} windows")

            msg = " · ".join(parts) if parts else "Not enough data."

            if label == "Low":
                st.success(f"Fragility: **Low** — {msg}")
            elif label == "Medium":
                st.warning(f"Fragility: **Medium** — {msg}")
            elif label == "High":
                st.error(f"Fragility: **High** — {msg}")
            else:
                st.info(f"Fragility: **—** — {msg}")

            # Detail plots (per-start windows)
            if rs_det is None or rs_det.empty:
                st.info("rolling_starts_detail.csv not found for this run yet. (You’ll still have the summary above.)")
            else:
                g = rs_det[rs_det["config_id"].astype(str) == str(inspect_id)].copy()
                if g.empty:
                    st.info("No per-start detail rows found for this config_id.")
                else:
                    # Parse & sort start date
                    if "start_dt" in g.columns:
                        g["start_dt"] = pd.to_datetime(g["start_dt"], errors="coerce")
                    if "start_i" in g.columns:
                        g["start_i"] = pd.to_numeric(g["start_i"], errors="coerce")
                    g = g.sort_values(["start_dt" if "start_dt" in g.columns else "start_i"])

                    # Normalize numeric fields
                    for c in ["performance.twr_total_return", "performance.max_drawdown_equity", "uw_max_days", "util_mean", "equity.net_profit_ex_cashflows"]:
                        if c in g.columns:
                            g[c] = pd.to_numeric(g[c], errors="coerce")

                    
                    # Quick pain points (what was the worst start date, and why?)
                    pains = []
                    try:
                        if "performance.twr_total_return" in g.columns:
                            rmin = g.dropna(subset=["performance.twr_total_return"]).nsmallest(1, "performance.twr_total_return").head(1)
                            if not rmin.empty:
                                rr = rmin.iloc[0]
                                s = rr.get("start_dt", rr.get("start_i", "—"))
                                pains.append(f"Worst return start: **{s}** → return {_fmt_pct(rr.get('performance.twr_total_return'))}, DD {_fmt_pct(rr.get('performance.max_drawdown_equity'))}, UW {int(0 if pd.isna(pd.to_numeric(rr.get('uw_max_days'), errors='coerce')) else pd.to_numeric(rr.get('uw_max_days'), errors='coerce'))}d")
                        if "performance.max_drawdown_equity" in g.columns:
                            dmax = g.dropna(subset=["performance.max_drawdown_equity"]).nlargest(1, "performance.max_drawdown_equity").head(1)
                            if not dmax.empty:
                                rr = dmax.iloc[0]
                                s = rr.get("start_dt", rr.get("start_i", "—"))
                                pains.append(f"Worst drawdown start: **{s}** → DD {_fmt_pct(rr.get('performance.max_drawdown_equity'))}, return {_fmt_pct(rr.get('performance.twr_total_return'))}")
                        if "uw_max_days" in g.columns:
                            umax = g.dropna(subset=["uw_max_days"]).nlargest(1, "uw_max_days").head(1)
                            if not umax.empty:
                                rr = umax.iloc[0]
                                s = rr.get("start_dt", rr.get("start_i", "—"))
                                pains.append(f"Longest underwater start: **{s}** → UW {int(0 if pd.isna(pd.to_numeric(rr.get('uw_max_days'), errors='coerce')) else pd.to_numeric(rr.get('uw_max_days'), errors='coerce'))}d, return {_fmt_pct(rr.get('performance.twr_total_return'))}")
                    except Exception:
                        pains = []

                    if pains:
                        st.markdown("**Worst-case highlights**  \\n" + "  \\n".join([f"- {p}" for p in pains]))

                    tabs = st.tabs(["Return vs start", "Drawdown vs start", "Underwater vs start", "Distributions", "Starts table"])

                    with tabs[0]:
                        if "performance.twr_total_return" not in g.columns:
                            st.info("Missing performance.twr_total_return in detail.")
                        else:
                            # Friendlier units for charts
                            if "performance.twr_total_return" in g.columns:
                                g["twr_pct"] = g["performance.twr_total_return"] * 100.0
                            if "performance.max_drawdown_equity" in g.columns:
                                g["dd_pct"] = g["performance.max_drawdown_equity"] * 100.0
                            fig_r = px.scatter(
                                g,
                                x=("start_dt" if "start_dt" in g.columns else "start_i"),
                                y="twr_pct",
                                hover_data=[c for c in ["start_dt", "start_i", "bars", "performance.max_drawdown_equity", "uw_max_days"] if c in g.columns],
                                labels={"twr_pct": "Total return (%)"},
                                title="Rolling Starts: return by start date",
                            )
                            # Add p10/p50/p90 reference lines from the summary row
                            for qname, dash in [("twr_p10", "dot"), ("twr_p50", "dash"), ("twr_p90", "dot")]:
                                if qname in row.columns and not pd.isna(r0.get(qname)):
                                    fig_r.add_hline(y=float(r0.get(qname)) * 100.0, line_dash=dash)
                            st.caption("Dotted lines = p10/p90. Dashed line = median (p50). The less these dots care about where you start, the more 'real' the edge is.")
                            _plotly(fig_r)

                            # Worst / best start dates quick peek
                            g_rank = g.dropna(subset=["performance.twr_total_return"]).copy()
                            if not g_rank.empty:
                                worst = g_rank.nsmallest(5, "performance.twr_total_return")
                                best = g_rank.nlargest(5, "performance.twr_total_return")
                                cc1, cc2 = st.columns(2)
                                with cc1:
                                    st.write("**Worst starts (by return)**")
                                    st.dataframe(
                                        worst[[c for c in ["start_dt", "start_i", "performance.twr_total_return", "performance.max_drawdown_equity", "uw_max_days"] if c in worst.columns]],
                                        width="stretch",
                                        height=210,
                                    )
                                with cc2:
                                    st.write("**Best starts (by return)**")
                                    st.dataframe(
                                        best[[c for c in ["start_dt", "start_i", "performance.twr_total_return", "performance.max_drawdown_equity", "uw_max_days"] if c in best.columns]],
                                        width="stretch",
                                        height=210,
                                    )

                    with tabs[1]:
                        if "performance.max_drawdown_equity" not in g.columns:
                            st.info("Missing performance.max_drawdown_equity in detail.")
                        else:
                            fig_d = px.scatter(
                                g,
                                x=("start_dt" if "start_dt" in g.columns else "start_i"),
                                y="dd_pct",
                                hover_data=[c for c in ["start_dt", "start_i", "bars", "performance.twr_total_return"] if c in g.columns],
                                labels={"dd_pct": "Max drawdown (%)"},
                                title="Rolling Starts: max drawdown by start date",
                            )
                            _plotly(fig_d)

                    with tabs[2]:
                        if "uw_max_days" not in g.columns:
                            st.info("Missing uw_max_days in detail.")
                        else:
                            fig_u = px.scatter(
                                g,
                                x=("start_dt" if "start_dt" in g.columns else "start_i"),
                                y="uw_max_days",
                                hover_data=[c for c in ["start_dt", "start_i", "bars", "performance.twr_total_return", "performance.max_drawdown_equity"] if c in g.columns],
                                labels={"uw_max_days": "Max underwater days"},
                                title="Rolling Starts: max underwater days by start date",
                            )
                            _plotly(fig_u)

                    
                    with tabs[3]:
                        # Distribution views help you see "how often does it hurt?"
                        cols_dist = st.columns(3)

                        if "performance.twr_total_return" in g.columns:
                            g["_twr_pct"] = pd.to_numeric(g["performance.twr_total_return"], errors="coerce") * 100.0
                            fig_hd = px.histogram(g.dropna(subset=["_twr_pct"]), x="_twr_pct", nbins=30, title="Return distribution (rolling starts)")
                            for qname, dash in [("twr_p10", "dot"), ("twr_p50", "dash"), ("twr_p90", "dot")]:
                                if qname in row.columns and not pd.isna(r0.get(qname)):
                                    fig_hd.add_vline(x=float(r0.get(qname)) * 100.0, line_dash=dash)
                            with cols_dist[0]:
                                _plotly(fig_hd)
                        else:
                            cols_dist[0].info("No return column in detail.")

                        if "performance.max_drawdown_equity" in g.columns:
                            g["_dd_pct"] = pd.to_numeric(g["performance.max_drawdown_equity"], errors="coerce") * 100.0
                            fig_dd = px.histogram(g.dropna(subset=["_dd_pct"]), x="_dd_pct", nbins=30, title="Drawdown distribution (rolling starts)")
                            for qname, dash in [("dd_p50", "dash"), ("dd_p90", "dot")]:
                                if qname in row.columns and not pd.isna(r0.get(qname)):
                                    fig_dd.add_vline(x=float(r0.get(qname)) * 100.0, line_dash=dash)
                            with cols_dist[1]:
                                _plotly(fig_dd)
                        else:
                            cols_dist[1].info("No drawdown column in detail.")

                        if "uw_max_days" in g.columns:
                            fig_uw = px.histogram(g.dropna(subset=["uw_max_days"]), x="uw_max_days", nbins=30, title="Underwater days distribution (rolling starts)")
                            for qname, dash in [("uw_p50_days", "dash"), ("uw_p90_days", "dot")]:
                                if qname in row.columns and not pd.isna(r0.get(qname)):
                                    fig_uw.add_vline(x=float(r0.get(qname)), line_dash=dash)
                            with cols_dist[2]:
                                _plotly(fig_uw)
                        else:
                            cols_dist[2].info("No underwater column in detail.")

                        st.caption("Dashed line = median. Dotted lines = p10/p90 (or p90 for drawdown/underwater). Tight distributions are what 'robust' looks like.")
                    with tabs[4]:
                        st.dataframe(g, width="stretch", height=420)

                    with st.expander("Exports (advanced)", expanded=False):
                        st.download_button(
                            "Download rolling-start view (CSV)",
                            data=df_show.to_csv(index=False).encode("utf-8"),
                            file_name=f"{selected_run_name}_rolling_view.csv",
                        )
                        st.download_button(
                            "Download rolling-start detail for this config (CSV)",
                            data=g.to_csv(index=False).encode("utf-8"),
                            file_name=f"{selected_run_name}_rs_detail_{inspect_id}.csv",
                        )

        with st.expander("Rolling Starts table", expanded=False):
            st.dataframe(df_show[cols], width="stretch", height=520)

        # Next step: Walkforward
        if st.button("Next: Walkforward →", type="primary", key="rs.next_to_wf"):
            st.session_state["ui.stage"] = "wf"
            st.rerun()

# Stage C: Walkforward
# =============================================================================

if stage_pick == "wf":
    st.write("### C) Walkforward (generalization)")
    st.caption("Splits the history into rolling windows and measures how performance behaves out-of-sample-ish.")

    # Walkforward availability
    wf_module_ok = True
    try:
        __import__("engine.walkforward")
    except Exception:
        wf_module_ok = False

    if not wf_module_ok:
        st.warning("Walkforward module not found/importable yet (engine.walkforward). UI wiring is ready though.")
        st.stop()


    # Choose WF run dir + parameters
    left, right = st.columns([2, 1])
    run_clicked = False

    with left:
        wf_runs = []
        if wf_root.exists():
            wf_runs = [p for p in wf_root.glob("wf_*") if p.is_dir()]
            wf_runs = sorted(wf_runs, key=lambda p: p.stat().st_mtime, reverse=True)

        wf_choice = st.selectbox(
            "Walkforward runs found",
            options=["(none)"] + [p.name for p in wf_runs],
            index=(1 if wf_runs else 0),
            key="wf.pick",
        )
        wf_dir = (wf_root / wf_choice) if (wf_choice != "(none)") else None

    # Build WF command in the sidebar-ish column, but RUN it full-width (below columns)
    cmd: Optional[List[str]] = None
    wf_progress: Optional[Path] = None

    with right:
        st.write("**Quick presets**")

        bars_per_day = _bars_per_day_from_run_meta(run_dir)
        bar_hint = _human_bar_interval_from_run(run_dir)
        st.caption(f"Detected timeframe: {bar_hint} (≈ {bars_per_day} bars/day)")

        preset = st.selectbox("Preset", options=["Quick", "Standard", "Thorough"], index=0, key="wf.preset")

        # Apply defaults only when preset changes (so number inputs don't reset constantly).
        prev = st.session_state.get("wf.preset_prev")
        if prev != preset:
            if bars_per_day <= 2:
                # Daily-ish: longer windows make sense
                if preset == "Quick":
                    w_default, s_default, cov = 180, 30, 0.90
                elif preset == "Standard":
                    w_default, s_default, cov = 365, 30, 0.90
                else:
                    w_default, s_default, cov = 730, 30, 0.90
            else:
                # Intraday: shorter calendar windows still contain many bars
                if preset == "Quick":
                    w_default, s_default, cov = 30, 7, 0.95
                elif preset == "Standard":
                    w_default, s_default, cov = 60, 7, 0.95
                else:
                    w_default, s_default, cov = 90, 3, 0.95

            expected = int(max(1, round(w_default * bars_per_day)))
            mb_default = int(max(1, math.ceil(expected * cov)))

            st.session_state["wf.window_days"] = int(w_default)
            st.session_state["wf.step_days"] = int(s_default)
            st.session_state["wf.min_bars"] = int(mb_default)
            st.session_state["wf.jobs"] = int(st.session_state.get("wf.jobs", 8))
            st.session_state["wf.preset_prev"] = preset

        # Avoid Streamlit warning: don't set both a widget default and session_state.
        if "wf.window_days" not in st.session_state:
            st.session_state["wf.window_days"] = int(365)
        if "wf.step_days" not in st.session_state:
            st.session_state["wf.step_days"] = int(30)

        window_days = int(st.number_input("Window days", min_value=7, max_value=3650, step=5, key="wf.window_days"))
        step_days = int(st.number_input("Step days", min_value=1, max_value=3650, step=5, key="wf.step_days"))

        expected_window_bars = int(max(1, round(window_days * bars_per_day)))
        st.caption(f"Expected bars per window: ~{expected_window_bars:,}. (Min bars must be ≤ this.)")

        max_mb = int(max(1, expected_window_bars))
        if "wf.min_bars" not in st.session_state:
            st.session_state["wf.min_bars"] = int(max_mb)
        # Clamp current value to widget bounds to avoid Streamlit exceptions.
        if int(st.session_state.get("wf.min_bars", 1)) > int(max_mb):
            st.session_state["wf.min_bars"] = int(max_mb)

        min_bars = int(st.number_input(
            "Min bars per window",
            min_value=1,
            max_value=max_mb,
            step=1,
            key="wf.min_bars",
        ))

        if "wf.jobs" not in st.session_state:
            st.session_state["wf.jobs"] = 8
        jobs = int(st.number_input("Jobs", min_value=1, max_value=64, step=1, key="wf.jobs"))

        survivors_ids = survivors["config_id"].astype(str).tolist()
        N = len(survivors_ids)

        # Clamp min_bars to something feasible for the chosen window
        expected_window_bars = int(max(1, round(window_days * bars_per_day))) if "bars_per_day" in locals() else int(window_days)
        min_bars_effective = int(min(int(min_bars), int(expected_window_bars)))
        if int(min_bars) != int(min_bars_effective):
            st.warning(f"Min bars ({min_bars}) exceeds expected bars/window (~{expected_window_bars}). Will clamp to {min_bars_effective}.")

        # WF output dir
        wf_out_dir = wf_root / f"wf_win{window_days}_step{step_days}_min{min_bars_effective}_n{N}"
        st.caption(f"Will run on survivors: {N} configs → output: {wf_out_dir}")

        run_clicked = st.button("Run Walkforward for all survivors", type="primary", disabled=(N == 0))

        if run_clicked:
            cmd = [
                PY,
                "-m",
                "engine.walkforward",
                "--from-run",
                str(run_dir),
                "--top-n",
                str(N),
                "--window-days",
                str(window_days),
                "--step-days",
                str(step_days),
                "--min-bars",
                str(min_bars_effective),
                "--jobs",
                str(jobs),
                "--out",
                str(wf_out_dir),
                "--sort-by",
                "gates.passed",  # stable, non-NaN, includes everyone selected by top-n
                "--sort-desc",
            ]
            wf_progress = wf_out_dir / "progress" / "walkforward.jsonl"
            wf_progress.parent.mkdir(parents=True, exist_ok=True)
            cmd += ["--no-progress", "--progress-file", str(wf_progress), "--progress-every", "25"]

    # Run full-width (NOT inside the right-side column) so the progress UI doesn't get squeezed.
    if run_clicked and cmd is not None and wf_progress is not None:
        st.markdown("---")
        try:
            _run_cmd(cmd, cwd=REPO_ROOT, label="Walkforward", progress_path=wf_progress)
            st.success("Walkforward complete.")
            st.rerun()
        except Exception as e:
            st.error(str(e))
            st.stop()

    wf_dir_effective = wf_dir or wf_out_dir
    wf_sum = load_wf_summary(wf_dir_effective)
    wf_rows = load_wf_results(wf_dir_effective)
    
    if wf_sum is None or wf_sum.empty:
        st.info("No walkforward stats found yet for the chosen output folder.")
        st.stop()
    
    base = merge_stage(survivors.copy(), wf_sum, on="config_id", suffix="wf")
    
    cov = int(base["wf.measured"].sum()) if "wf.measured" in base.columns else 0
    st.success(f"Coverage: {cov}/{len(base)} configs have walkforward stats in this folder.")
    
    with st.expander("Walkforward questions (filters)", expanded=True):
        wf_ans = _question_ui(walkforward_questions(), key_prefix="q.wf")
    
    dfC = apply_stage_eval(base, stage_key="wfq", questions=walkforward_questions(), answers=wf_ans)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        show_pass = st.checkbox("Show PASS", value=True, key="f.wf.pass")
    with col2:
        show_warn = st.checkbox("Show WARN", value=True, key="f.wf.warn")
    with col3:
        show_fail = st.checkbox("Show FAIL", value=False, key="f.wf.fail")
    
    keep = []
    if show_pass:
        keep.append("PASS")
    if show_warn:
        keep.append("WARN")
    if show_fail:
        keep.append("FAIL")
    df_show = dfC[dfC["wfq.verdict"].isin(keep)].copy()
    
    
    cols = [
        "config_id",
        "config.label",
        "wfq.verdict",
        "return_p10",
        "return_p50",
        "return_p90",
        "dd_p90",
        "uw_days_p90",
        "pct_profitable_windows",
        "pct_windows_traded",
        "trades_p10",
        "trades_p50",
        "min_window_return",
        "median_window_return",
        "stitched_total_return",
        "stitched_max_drawdown",
        "windows",
    ]
    cols = [c for c in cols if c in df_show.columns]
    if "wf.measured" in df_show.columns:
        cols.insert(2, "wf.measured")
    st.dataframe(df_show[cols], width="stretch", height=520)

    st.download_button(
        "Download walkforward view (CSV)",
        data=df_show.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_run_name}_walkforward_view.csv",
    )

    st.markdown("---")
    st.write("### Inspect a strategy (Walkforward)")

    with st.expander("How to read Walkforward (what these charts mean)", expanded=True):
        st.write("Walkforward chops history into many rolling windows (episodes). Each dot in the charts is one episode.")
        st.write("You're looking for consistency across episodes — not one lucky stretch.")
        st.write("• Return over time: each dot is total return inside one window. Tight clusters beat wild scatter.")
        st.write("• Drawdown over time: each dot is the worst peak→trough drop inside that window. Spikes mean occasional pain.")
        st.write("• Underwater days: how long equity stayed below its prior peak inside the window (recovery time).")
        st.write("• Histogram: p10/p50/p90 are worst-typical / typical / best-typical window outcomes.")
        st.write("• Stitched curve: compounds non-overlapping step slices to avoid overlap. It's a stability visualization, not a promise of tradability.")


    if wf_rows is None or wf_rows.empty:
        st.info("No per-window walkforward rows found yet (wf_results.csv).")
    else:
        # Pick a config to inspect (default: highest typical return)
        if "return_p50" in df_show.columns:
            opts = (
                df_show.sort_values("return_p50", ascending=False)["config_id"].astype(str).tolist()
            )
        else:
            opts = df_show["config_id"].astype(str).tolist()

        if not opts:
            st.info("No configs in the current filter set.")
        else:
            pick_id = st.selectbox("Config", options=opts, index=0, key="wf.inspect.pick")

            wsub = wf_rows[wf_rows["config_id"].astype(str) == str(pick_id)].copy()
            wsub = wsub.sort_values("window_idx", kind="mergesort")

            sum_row = None
            try:
                ssub = wf_sum[wf_sum["config_id"].astype(str) == str(pick_id)]
                if ssub is not None and not ssub.empty:
                    sum_row = ssub.iloc[0].to_dict()
            except Exception:
                sum_row = None

            # Summary metrics
            if sum_row:
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("WF p50 return", _fmt_pct(sum_row.get("return_p50")))
                with m2:
                    st.metric("WF p10 return", _fmt_pct(sum_row.get("return_p10")))
                with m3:
                    st.metric("WF dd_p90", _fmt_pct(sum_row.get("dd_p90")))
                with m4:
                    st.metric("% profitable windows", _fmt_pct(sum_row.get("pct_profitable_windows")))

            # Per-window timeline
            if "flags" in wsub.columns:
                wsub["has_flags"] = wsub["flags"].astype(str).str.len() > 0
            else:
                wsub["has_flags"] = False

            if "window_start_dt" in wsub.columns:
                x = "window_start_dt"
            else:
                x = "window_idx"

            yret = "window_return" if "window_return" in wsub.columns else "equity.total_return"
            fig = px.scatter(
                wsub,
                x=x,
                y=yret,
                color="has_flags",
                hover_data=[c for c in ["window_end_dt", "window_max_drawdown", "window_underwater_days", "trades_closed", "flags"] if c in wsub.columns],
                title="Walkforward windows: return over time",
            )
            # Quantile guides: teach the user to think in distributions (bad/typical/good windows).
            try:
                _vals = pd.to_numeric(wsub[yret], errors="coerce").dropna()
                if not _vals.empty:
                    r10 = float(_vals.quantile(0.10))
                    r50 = float(_vals.quantile(0.50))
                    r90 = float(_vals.quantile(0.90))
                    fig.add_hline(y=r50, line_dash="dash", annotation_text=f"p50 {r50:.1%}", annotation_position="top left")
                    fig.add_hline(y=r10, line_dash="dot", annotation_text=f"p10 {r10:.1%}", annotation_position="bottom left")
                    fig.add_hline(y=r90, line_dash="dot", annotation_text=f"p90 {r90:.1%}", annotation_position="top left")
            except Exception:
                pass
            fig.update_yaxes(tickformat=".0%")
            _plotly(fig)
            st.caption("Each dot is one window. Tight clusters beat lucky spikes. p10/p50/p90 lines show bad/typical/good window outcomes.")

            if "window_max_drawdown" in wsub.columns:
                fig2 = px.scatter(
                    wsub,
                    x=x,
                    y="window_max_drawdown",
                    color="has_flags",
                    hover_data=[c for c in ["window_return", "window_underwater_days", "trades_closed", "flags"] if c in wsub.columns],
                    title="Walkforward windows: max drawdown over time",
                )
                try:
                    d90 = float(pd.to_numeric(wsub["window_max_drawdown"], errors="coerce").dropna().quantile(0.90))
                    fig2.add_hline(y=d90, line_dash="dash", annotation_text=f"dd_p90 {d90:.1%}", annotation_position="top left")
                except Exception:
                    pass
                fig2.update_yaxes(tickformat=".0%")
                _plotly(fig2)
                st.caption("Each dot is the worst peak→trough drop inside that window. dd_p90 is your 'bad but typical' drawdown anchor.")

            if "window_underwater_days" in wsub.columns:
                fig_uw = px.scatter(
                    wsub,
                    x=x,
                    y="window_underwater_days",
                    color="has_flags",
                    hover_data=[c for c in ["window_return", "window_max_drawdown", "trades_closed", "flags"] if c in wsub.columns],
                    title="Walkforward windows: underwater days over time",
                )
                try:
                    uw90 = float(pd.to_numeric(wsub["window_underwater_days"], errors="coerce").dropna().quantile(0.90))
                    fig_uw.add_hline(y=uw90, line_dash="dash", annotation_text=f"uw_p90 {uw90:.0f}d", annotation_position="top left")
                except Exception:
                    pass
                _plotly(fig_uw)
                st.caption("Underwater days = time spent below the previous equity peak inside the window. High values mean long recovery / long boredom.")

            # Distribution (sanity check)
            if "window_return" in wsub.columns:
                fig3 = px.histogram(wsub, x="window_return", nbins=30, title="Window return distribution")
                try:
                    _vals = pd.to_numeric(wsub["window_return"], errors="coerce").dropna()
                    if not _vals.empty:
                        r10 = float(_vals.quantile(0.10))
                        r50 = float(_vals.quantile(0.50))
                        r90 = float(_vals.quantile(0.90))
                        fig3.add_vline(x=r50, line_dash="dash", annotation_text=f"p50 {r50:.1%}", annotation_position="top")
                        fig3.add_vline(x=r10, line_dash="dot", annotation_text=f"p10 {r10:.1%}", annotation_position="top")
                        fig3.add_vline(x=r90, line_dash="dot", annotation_text=f"p90 {r90:.1%}", annotation_position="top")
                except Exception:
                    pass
                fig3.update_xaxes(tickformat=".0%")
                _plotly(fig3)
                st.caption("Histogram of window returns. p10 is the 'worst-typical' anchor; p50 is typical; p90 is best-typical.")

            # Window leaderboard (failure modes)
            if "window_return" in wsub.columns:
                st.write("**Window leaderboard (failure modes)**")
                show_cols = [c for c in ["window_idx", "window_start_dt", "window_end_dt", "window_return", "window_max_drawdown", "window_underwater_days", "trades_closed", "flags"] if c in wsub.columns]

                t1, t2, t3 = st.tabs(["Worst return", "Worst drawdown", "Longest underwater"])
                with t1:
                    st.dataframe(
                        wsub.sort_values("window_return", ascending=True)[show_cols].head(10),
                        width="stretch",
                        height=260,
                    )
                with t2:
                    if "window_max_drawdown" in wsub.columns:
                        st.dataframe(
                            wsub.sort_values("window_max_drawdown", ascending=False)[show_cols].head(10),
                            width="stretch",
                            height=260,
                        )
                    else:
                        st.info("No drawdown column for this walkforward run.")
                with t3:
                    if "window_underwater_days" in wsub.columns:
                        st.dataframe(
                            wsub.sort_values("window_underwater_days", ascending=False)[show_cols].head(10),
                            width="stretch",
                            height=260,
                        )
                    else:
                        st.info("No underwater-days column for this walkforward run.")

            # Stitched curve (non-overlapping segments)
            stitched_path = None
            try:
                if sum_row and sum_row.get("stitched_path"):
                    stitched_path = wf_dir_effective / str(sum_row["stitched_path"])
                else:
                    stitched_path = wf_dir_effective / "stitched" / f"{pick_id}.csv"
            except Exception:
                stitched_path = wf_dir_effective / "stitched" / f"{pick_id}.csv"

            if stitched_path is not None and stitched_path.exists():
                st.write("**Stitched curve (non-overlapping segments)**")
                st.caption("This compounds step-sized slices to avoid overlap. It's a stability visualization, not a promise of tradability.")
                sdf = _load_csv(stitched_path)
                if sdf is not None and not sdf.empty and "stitched_twr" in sdf.columns:
                    fig4 = px.line(sdf, x="dt" if "dt" in sdf.columns else sdf.columns[0], y="stitched_twr", title="Stitched TWR index")
                    try:
                        fig4.add_hline(y=1.0, line_dash="dash", annotation_text="start (1.0)", annotation_position="bottom right")
                    except Exception:
                        pass
                    _plotly(fig4)

                    st.download_button(
                        "Download stitched curve (CSV)",
                        data=sdf.to_csv(index=False).encode("utf-8"),
                        file_name=f"{selected_run_name}_wf_stitched_{pick_id}.csv",
                    )
            else:
                st.info("No stitched curve found for this config (expected under wf_dir/stitched/).")

            # Downloads
            st.download_button(
                "Download per-window rows for this config (CSV)",
                data=wsub.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected_run_name}_wf_windows_{pick_id}.csv",
            )

    
    # =============================================================================

# =============================================================================
# Stage D: Grand verdict + deep dive
# =============================================================================

def _apply_grand_preset(preset: str) -> None:
    """
    Presets are *just defaults* for the question radios in Grand Verdict.
    They exist to teach the "target profile" habit: pick pain limits first, then filter.
    """
    preset = (preset or "").strip().lower()
    if preset in {"", "none"}:
        return

    # Choice indexes correspond to the order inside each QuestionSpec.
    PRESETS: Dict[str, Dict[str, Dict[str, int]]] = {
        # Tight pain limits. Fewer survivors, more stability.
        "conservative": {
            "batch": {"batch_drawdown": 0, "batch_profit": 0, "batch_fees": 0},
            "rs": {"rs_worst_return": 0, "rs_drawdown": 0, "rs_underwater": 0, "rs_util": 0},
            "wf": {"wf_typical": 0, "wf_worst_typical": 0, "wf_min": 0, "wf_dd": 0, "wf_consistency": 0, "wf_trading": 0},
        },
        # "Reasonable adult" defaults. Lets exploration happen without being naive.
        "balanced": {
            "batch": {"batch_drawdown": 1, "batch_profit": 0, "batch_fees": 1},
            "rs": {"rs_worst_return": 1, "rs_drawdown": 1, "rs_underwater": 2, "rs_util": 1},
            "wf": {"wf_typical": 0, "wf_worst_typical": 1, "wf_min": 1, "wf_dd": 1, "wf_consistency": 1, "wf_trading": 1},
        },
        # Loose filters. Useful early when you want to see "what exists", not only survivors.
        "aggressive": {
            "batch": {"batch_drawdown": 2, "batch_profit": 0, "batch_fees": 2},
            "rs": {"rs_worst_return": 2, "rs_drawdown": 2, "rs_underwater": 3, "rs_util": 2},
            "wf": {"wf_typical": 0, "wf_worst_typical": 2, "wf_min": 2, "wf_dd": 2, "wf_consistency": 2, "wf_trading": 2},
        },
    }

    p = PRESETS.get(preset)
    if not p:
        return

    # Keys used by _question_ui
    for qid, idx in p["batch"].items():
        st.session_state[f"q.grand.batch.{qid}"] = int(idx)
    for qid, idx in p["rs"].items():
        st.session_state[f"q.grand.rs.{qid}"] = int(idx)
    for qid, idx in p["wf"].items():
        st.session_state[f"q.grand.wf.{qid}"] = int(idx)


def _grand_score_row(r: Dict[str, Any]) -> float:
    """
    Worst-case-aware score. Higher is better.

    Uses:
      - WF return_p10 and dd_p90 (most important)
      - RS twr_p10 and dd_p90 (second)
      - Batch TWR return and drawdown (third)

    Underwater days are converted to "years" to keep units sane.
    """
    wf_r10 = _to_float(r.get("return_p10", float("nan")))
    wf_dd90 = _to_float(r.get("dd_p90", float("nan")))
    wf_uw90 = _to_float(r.get("uw_days_p90", float("nan")))

    rs_r10 = _to_float(r.get("twr_p10", float("nan")))
    rs_dd90 = _to_float(r.get("dd_p90", float("nan")))  # RS uses dd_p90 too
    rs_uw90 = _to_float(r.get("uw_p90_days", float("nan")))

    b_r = _to_float(r.get("performance.twr_total_return", float("nan")))
    b_dd = _to_float(r.get("performance.max_drawdown_equity", float("nan")))

    def _nan0(x: float) -> float:
        return 0.0 if (x != x) else float(x)

    wf = _nan0(wf_r10) - 0.50 * _nan0(wf_dd90) - 0.10 * (_nan0(wf_uw90) / 365.0)
    rs = _nan0(rs_r10) - 0.50 * _nan0(rs_dd90) - 0.10 * (_nan0(rs_uw90) / 365.0)
    bt = _nan0(b_r) - 0.50 * _nan0(b_dd)

    # Penalize missing measurements (unmeasured WF/RS shouldn't float to the top).
    wf_pen = 0.25 if (wf_r10 != wf_r10 or wf_dd90 != wf_dd90) else 0.0
    rs_pen = 0.10 if (rs_r10 != rs_r10 or rs_dd90 != rs_dd90) else 0.0

    return 0.60 * wf + 0.30 * rs + 0.10 * bt - wf_pen - rs_pen


if stage_pick == "grand":
    # -------------------------------------------------------------------------
    # Cockpit view (MVP): preferences → shortlist → evidence
    # -------------------------------------------------------------------------
    st.subheader("Cockpit")
    st.caption("Define your pain limits first → then pick survivors → then inspect evidence. Results are view-only.")

    # Load latest RS/WF if present
    rs_dir_effective = rs_latest
    wf_dir_effective = wf_latest

    rs_sum = load_rs_summary(run_dir, rs_dir_effective) if rs_dir_effective else None
    wf_sum = load_wf_summary(wf_dir_effective) if wf_dir_effective else None

    df = survivors.copy()
    df = _ensure_config_id(df)

    # =========================
    # Preferences wedge
    # =========================
    with st.container(border=True):
        st.markdown("#### Preferences (your wedge)")

        preset = st.selectbox(
            "Target profile preset",
            options=["Balanced", "Conservative", "Aggressive", "Custom"],
            index=0,
            key="grand.profile_preset_v2",
            help="Presets set sane defaults for the limit radios below. 'Custom' keeps your current choices.",
        )

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("Apply preset", key="grand.apply_preset_btn_v2", disabled=(preset == "Custom")):
                # Capture current answers → apply preset → compute diff (so users can see what changed)
                bqs = batch_questions()
                rqs = rolling_questions()
                wqs = walkforward_questions()

                def _get_idx(prefix: str, q) -> int:
                    key = f"{prefix}.{q.id}"
                    try:
                        return int(st.session_state.get(key, int(q.default_index)))
                    except Exception:
                        return int(getattr(q, "default_index", 0) or 0)

                def _choice_label(q, idx: int) -> str:
                    try:
                        idx2 = max(0, min(int(idx), len(q.choices) - 1))
                        return str(q.choices[idx2].label)
                    except Exception:
                        return str(idx)

                before = {}
                for q in bqs:
                    before[("Batch", q.id)] = _get_idx("q.grand.batch", q)
                for q in rqs:
                    before[("Rolling Starts", q.id)] = _get_idx("q.grand.rs", q)
                for q in wqs:
                    before[("Walkforward", q.id)] = _get_idx("q.grand.wf", q)

                _apply_grand_preset(str(preset))

                changes = []
                for q in bqs:
                    a = _get_idx("q.grand.batch", q)
                    b = int(before.get(("Batch", q.id), a))
                    if a != b:
                        changes.append(("Batch", str(getattr(q, "title", getattr(q, "id", ""))), _choice_label(q, b), _choice_label(q, a)))
                for q in rqs:
                    a = _get_idx("q.grand.rs", q)
                    b = int(before.get(("Rolling Starts", q.id), a))
                    if a != b:
                        changes.append(("Rolling Starts", str(getattr(q, "title", getattr(q, "id", ""))), _choice_label(q, b), _choice_label(q, a)))
                for q in wqs:
                    a = _get_idx("q.grand.wf", q)
                    b = int(before.get(("Walkforward", q.id), a))
                    if a != b:
                        changes.append(("Walkforward", str(getattr(q, "title", getattr(q, "id", ""))), _choice_label(q, b), _choice_label(q, a)))

                st.session_state["grand.last_preset_applied"] = str(preset)
                st.session_state["grand.last_preset_changes"] = list(changes)
        with c2:
            show_help = st.checkbox("Show explainer", value=False, key="grand.show_help")
        with c3:
            st.caption("Presets only change filters. They never modify your data or rerun anything.")


        last_preset = st.session_state.get("grand.last_preset_applied")
        last_changes = st.session_state.get("grand.last_preset_changes")
        if last_preset and isinstance(last_changes, list):
            if len(last_changes) == 0:
                st.info(f"Preset **{last_preset}** matched your current limits (no changes).")
            else:
                st.success(f"Applied preset **{last_preset}** → updated {len(last_changes)} limit choices.")
                with st.expander("Show what the preset changed", expanded=True):
                    # Keep it readable: show the first N changes.
                    for section, q_label, old_label, new_label in last_changes[:40]:
                        st.write(f"**{section}** · {q_label}: `{old_label}` → `{new_label}`")
                    if len(last_changes) > 40:
                        st.caption(f"(Showing 40 of {len(last_changes)} changes.)")

        if show_help:
            st.markdown('''
- **PASS / WARN / FAIL** are driven by the limit radios below.  
- **PASS** = within limits. **WARN** = suspicious but maybe acceptable. **FAIL** = exceeds hard limits.  
- If Rolling Starts / Walkforward are **missing**, either ignore them (early exploration) or require them (trust mode).
'''.strip())

        # Limit radios (collapsed by default; the preset sets defaults)
        with st.expander("Batch limits", expanded=True):
            batch_ans = _question_ui(batch_questions(), key_prefix="q.grand.batch")
        df = apply_stage_eval(df, stage_key="batch", questions=batch_questions(), answers=batch_ans)

        rs_ans: Dict[str, int] = {}
        if rs_sum is not None and not rs_sum.empty:
            df = merge_stage(df, rs_sum, on="config_id", suffix="rs")
            with st.expander("Rolling Starts limits", expanded=True):
                rs_ans = _question_ui(rolling_questions(), key_prefix="q.grand.rs")
            df = apply_stage_eval(df, stage_key="rsq", questions=rolling_questions(), answers=rs_ans)
        else:
            df["rs.measured"] = False
            df["rsq.verdict"] = "UNMEASURED"

        wf_ans: Dict[str, int] = {}
        if wf_sum is not None and not wf_sum.empty:
            df = merge_stage(df, wf_sum, on="config_id", suffix="wf")
            with st.expander("Walkforward limits", expanded=True):
                wf_ans = _question_ui(walkforward_questions(), key_prefix="q.grand.wf")
            df = apply_stage_eval(df, stage_key="wfq", questions=walkforward_questions(), answers=wf_ans)
        else:
            df["wf.measured"] = False
            df["wfq.verdict"] = "UNMEASURED"

        st.divider()

        col1, col2, col3 = st.columns(3)
        with col1:
            req_batch = st.selectbox("Require Batch", options=["PASS only", "PASS or WARN", "Ignore"], index=1, key="grand.req_batch")
        with col2:
            req_rs = st.selectbox("Require Rolling Starts", options=["PASS only", "PASS or WARN", "Ignore"], index=1, key="grand.req_rs")
        with col3:
            req_wf = st.selectbox("Require Walkforward", options=["PASS only", "PASS or WARN", "Ignore"], index=1, key="grand.req_wf")

        # Verdict visibility toggles (global)
        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            show_pass = st.checkbox("Show PASS", value=True, key="grand.show_pass")
        with vc2:
            show_warn = st.checkbox("Show WARN", value=True, key="grand.show_warn")
        with vc3:
            show_fail = st.checkbox("Show FAIL/UNMEASURED", value=False, key="grand.show_fail")

        st.markdown("#### Ranking")
        st.caption("The score is a ranking hint. The evidence tabs are the receipts.")

    # =========================
    # Build the shortlist (unified candidates table)
    # =========================
    def _keep(verdict: str, rule: str) -> bool:
        if rule.startswith("Ignore"):
            return True
        if rule.startswith("PASS only"):
            return verdict == "PASS"
        return verdict in {"PASS", "WARN"}

    keep_mask: List[bool] = []
    grand_verdicts: List[str] = []

    for _, r in df.iterrows():
        ok = True
        stage_vs: List[str] = []

        v_batch = str(r.get("batch.verdict", ""))
        ok = ok and _keep(v_batch, req_batch)
        if not req_batch.startswith("Ignore"):
            stage_vs.append(v_batch)

        v_rs = str(r.get("rsq.verdict", "UNMEASURED"))
        if v_rs == "UNMEASURED":
            ok = ok and (req_rs == "Ignore")
            if not req_rs.startswith("Ignore"):
                stage_vs.append("UNMEASURED")
        else:
            ok = ok and _keep(v_rs, req_rs)
            if not req_rs.startswith("Ignore"):
                stage_vs.append(v_rs)

        v_wf = str(r.get("wfq.verdict", "UNMEASURED"))
        if v_wf == "UNMEASURED":
            ok = ok and (req_wf == "Ignore")
            if not req_wf.startswith("Ignore"):
                stage_vs.append("UNMEASURED")
        else:
            ok = ok and _keep(v_wf, req_wf)
            if not req_wf.startswith("Ignore"):
                stage_vs.append(v_wf)

        keep_mask.append(bool(ok))

        if "FAIL" in stage_vs or "UNMEASURED" in stage_vs:
            gv = "FAIL" if "UNMEASURED" not in stage_vs else "UNMEASURED"
        elif "WARN" in stage_vs:
            gv = "WARN"
        else:
            gv = "PASS"
        grand_verdicts.append(gv)

    df["grand.verdict"] = grand_verdicts
    df2 = df[pd.Series(keep_mask, index=df.index)].copy()

    if not df2.empty:
        df2["score.grand_robust"] = [_grand_score_row(r) for r in df2.to_dict(orient="records")]
        df2["score.grand_robust"] = pd.to_numeric(df2["score.grand_robust"], errors="coerce")

    sort_opts: List[str] = []
    for c in [
        "score.grand_robust",
        "score.calmar_equity",
        "robustness_score",
        "return_p10",
        "return_p50",
        "dd_p90",
        "pct_profitable_windows",
        "pct_windows_traded",
        "twr_p10",
        "twr_p50",
        "performance.twr_total_return",
        "performance.max_drawdown_equity",
        "equity.net_profit_ex_cashflows",
    ]:
        if c in df2.columns and c not in sort_opts:
            sort_opts.append(c)
    if not sort_opts:
        sort_opts = ["config_id"]

    sort_by = st.selectbox("Sort by", options=sort_opts, index=0, key="grand.sort_by")
    ascending = st.checkbox("Ascending", value=False, key="grand.asc")
    if sort_by in df2.columns and not df2.empty:
        df2[sort_by] = pd.to_numeric(df2[sort_by], errors="coerce")
        df2 = df2.sort_values(sort_by, ascending=bool(ascending))

    mask_v: List[bool] = []
    for v in df2.get("grand.verdict", pd.Series([], dtype=str)).astype(str):
        if v == "PASS" and show_pass:
            mask_v.append(True)
        elif v == "WARN" and show_warn:
            mask_v.append(True)
        elif v in {"FAIL", "UNMEASURED"} and show_fail:
            mask_v.append(True)
        else:
            mask_v.append(False)

    df_show = df2[pd.Series(mask_v, index=df2.index)] if (len(mask_v) == len(df2)) else df2


    # =========================
    # Run story (population-level explainability)
    # =========================
    with st.container(border=True):
        st.markdown("#### Run story (what happened to the whole run)")
        st.caption("After your preferences are applied, these charts summarize how the *population* of strategies behaved — not just one config.")

        show_story = st.checkbox("Show run story charts", value=True, key="story.show")
        if not show_story:
            st.caption("Run story hidden.")
        elif px is None or go is None:
            st.info("Plotly is not available in this environment, so charts are disabled.")
        else:
            df_all = df.copy()
            df_req = df2.copy()
            df_vis = df_show.copy()

            # Funnel: evaluated → meets requirements → visible
            funnel = pd.DataFrame(
                {
                    "Stage": ["Survivors evaluated", "Meets requirements", "Visible now"],
                    "Count": [int(len(df_all)), int(len(df_req)), int(len(df_vis))],
                }
            )
            fig_funnel = go.Figure(
                go.Funnel(
                    y=funnel["Stage"],
                    x=funnel["Count"],
                    textinfo="value+percent initial",
                    marker=dict(color=[ACCENT_BLUE, WARN_COLOR, PASS_COLOR]),
                )
            )
            _style_fig(fig_funnel, title="Survivor funnel")

            # Verdict distribution by stage (stacked)
            rows = []
            for stage_label, col in [
                ("Batch", "batch.verdict"),
                ("Rolling Starts", "rsq.verdict"),
                ("Walkforward", "wfq.verdict"),
                ("Grand", "grand.verdict"),
            ]:
                if col in df_all.columns:
                    vc = df_all[col].fillna("UNMEASURED").astype(str).value_counts()
                    for v, cnt in vc.items():
                        rows.append({"Stage": stage_label, "Verdict": str(v), "Count": int(cnt)})
            df_stage = pd.DataFrame(rows)

            fig_stage = None
            if not df_stage.empty:
                fig_stage = px.bar(
                    df_stage,
                    x="Stage",
                    y="Count",
                    color="Verdict",
                    barmode="stack",
                    color_discrete_map={k: _verdict_color(k) for k in df_stage["Verdict"].unique()},
                )
                _style_fig(fig_stage, title="Verdict mix by stage")
                fig_stage.update_layout(xaxis_title=None, yaxis_title=None)

            c1, c2 = st.columns([1.0, 1.2])
            with c1:
                _plotly(fig_funnel)
            with c2:
                if fig_stage is not None:
                    _plotly(fig_stage)
                else:
                    st.caption("No verdict columns found to build stage distribution.")

                        # Risk/return map (drawdown vs return), highlight the currently-visible shortlist
            dd_col = _pick_col(df_all, ["performance.max_drawdown_equity", "performance.max_drawdown", "equity.max_drawdown"])
            ret_col = _pick_col(df_all, ["performance.twr_total_return", "equity.net_profit_ex_cashflows", "equity.net_profit"])
            if dd_col and ret_col and dd_col in df_all.columns and ret_col in df_all.columns and go is not None:
                base = df_all.copy()
                base["_dd"] = _drawdown_to_frac(pd.to_numeric(base[dd_col], errors="coerce"))
                base["_ret"] = pd.to_numeric(base[ret_col], errors="coerce")
                base = base.dropna(subset=["_dd", "_ret"])

                hi = df_vis.copy()
                if dd_col in hi.columns and ret_col in hi.columns:
                    hi["_dd"] = _drawdown_to_frac(pd.to_numeric(hi[dd_col], errors="coerce"))
                    hi["_ret"] = pd.to_numeric(hi[ret_col], errors="coerce")
                    hi = hi.dropna(subset=["_dd", "_ret"])
                else:
                    hi = pd.DataFrame()

                # Robustness percentile within the currently-visible shortlist (if score exists)
                score_col = None
                for c in ["score.grand_robust", "robustness_score"]:
                    if c in hi.columns:
                        score_col = c
                        break
                robust_pct = {}
                if score_col and "config_id" in hi.columns:
                    try:
                        _s = pd.to_numeric(hi[score_col], errors="coerce")
                        _r = _s.rank(pct=True, ascending=True)
                        robust_pct = {str(cid): float(p) for cid, p in zip(hi["config_id"].astype(str), _r)}
                    except Exception:
                        robust_pct = {}

                def _verdict_symbol(v: str) -> str:
                    vv = str(v or "").upper()
                    if vv == "PASS":
                        return "circle"
                    if vv == "WARN":
                        return "triangle-up"
                    if vv in {"FAIL", "UNMEASURED"}:
                        return "x"
                    return "diamond"

                def _reason_snippet(row_dict: Dict[str, Any]) -> str:
                    """One-line reason snippet (first/highest-severity violation under current prefs)."""
                    sev_rank = {"critical": 0, "warn": 1, "info": 2}
                    best = None  # (sev, stage, msg)
                    try:
                        out_b = evaluate_row_with_questions(row_dict, batch_questions(), batch_ans)
                        for v in getattr(out_b, "violations", []) or []:
                            r = sev_rank.get(str(v.get("severity", "info")).lower(), 9)
                            msg = str(v.get("message", "")).strip()
                            if msg and (best is None or r < best[0]):
                                best = (r, "Batch", msg)
                    except Exception:
                        pass
                    try:
                        if rs_sum is not None and not rs_sum.empty:
                            out_r = evaluate_row_with_questions(row_dict, rolling_questions(), rs_ans)
                            for v in getattr(out_r, "violations", []) or []:
                                r = sev_rank.get(str(v.get("severity", "info")).lower(), 9)
                                msg = str(v.get("message", "")).strip()
                                if msg and (best is None or r < best[0]):
                                    best = (r, "RS", msg)
                    except Exception:
                        pass
                    try:
                        if wf_sum is not None and not wf_sum.empty:
                            out_w = evaluate_row_with_questions(row_dict, walkforward_questions(), wf_ans)
                            for v in getattr(out_w, "violations", []) or []:
                                r = sev_rank.get(str(v.get("severity", "info")).lower(), 9)
                                msg = str(v.get("message", "")).strip()
                                if msg and (best is None or r < best[0]):
                                    best = (r, "WF", msg)
                    except Exception:
                        pass
                    if best is None:
                        return "No violations under current preferences."
                    msg = best[2]
                    if len(msg) > 120:
                        msg = msg[:117].rstrip() + "…"
                    return f"{best[1]}: {msg}"

                # Axis formatting depends on whether we're showing a return fraction or an absolute profit ($)
                is_currency = ("net_profit" in str(ret_col).lower()) or ("profit" in str(ret_col).lower() and "return" not in str(ret_col).lower())

                fig_rr = go.Figure()
                fig_rr.add_trace(
                    go.Scattergl(
                        x=base["_dd"],
                        y=base["_ret"],
                        mode="markers",
                        name="All survivors (faint)",
                        marker=dict(size=6, color="rgba(17,24,39,0.12)", symbol="circle"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

                if not hi.empty and "grand.verdict" in hi.columns:
                    # Precompute hover fields for visible shortlist
                    hi2 = hi.copy()
                    hi2["_cid"] = hi2["config_id"].astype(str) if "config_id" in hi2.columns else ""
                    label_col = _pick_col(hi2, ["config.label", "label", "config_label"])
                    hi2["_label"] = hi2[label_col].astype(str) if label_col and label_col in hi2.columns else ""
                    # Normalize verdicts; keep PASS/WARN/FAIL/UNMEASURED, collapse everything else to OTHER (hidden in legend)
                    _allowed = {"PASS", "WARN", "FAIL", "UNMEASURED"}
                    hi2["_grand_raw"] = hi2["grand.verdict"].astype(str)
                    hi2["_grand"] = hi2["_grand_raw"].str.upper()
                    hi2["_grand"] = hi2["_grand"].where(hi2["_grand"].isin(_allowed), "OTHER")
                    hi2["_rob"] = hi2["_cid"].map(robust_pct)
                    hi2["_rob_str"] = hi2["_rob"].apply(lambda p: f"{int(round(float(p)*100))}th pct" if pd.notna(p) else "—")
                    # Reason snippet (small N; OK to compute per-row)
                    try:
                        hi2["_reason"] = [ _reason_snippet(r) for r in hi2.to_dict("records") ]
                    except Exception:
                        hi2["_reason"] = ""

                    order = ["PASS", "WARN", "FAIL", "UNMEASURED", "OTHER"]
                    for v in order:
                        g = hi2[hi2["_grand"].astype(str) == str(v)]
                        if g.empty:
                            continue
                        custom = np.stack(
                            [
                                g["_cid"].to_numpy(),
                                g["_label"].to_numpy(),
                                g["_grand"].to_numpy(),
                                g["_rob_str"].to_numpy(),
                                g["_reason"].to_numpy(),
                            ],
                            axis=1,
                        )
                        hover_ret = "%{y:.2%}" if not is_currency else "$%{y:,.0f}"
                        fig_rr.add_trace(
                            go.Scattergl(
                                x=g["_dd"],
                                y=g["_ret"],
                                mode="markers",
                                name=("Other" if str(v) == "OTHER" else str(v)),
                                showlegend=(str(v) != "OTHER"),
                                marker=dict(
                                    size=11,
                                    color=_verdict_color(v),
                                    symbol=_verdict_symbol(v),
                                    opacity=0.95,
                                    line=dict(width=0.8, color="rgba(17,24,39,0.35)"),
                                ),
                                customdata=custom,
                                hovertemplate=(
                                    "config=%{customdata[0]}<br>"
                                    "label=%{customdata[1]}<br>"
                                    "grand verdict=%{customdata[2]}<br>"
                                    "robustness=%{customdata[3]}<br>"
                                    "max DD=%{x:.2%}<br>"
                                    f"return={hover_ret}<br>"
                                    "reason=%{customdata[4]}<extra></extra>"
                                ),
                            )
                        )

                st.markdown("#### Risk/return map (population view)")
                st.caption("Faint = all survivors · Bold = visible shortlist · Markers: PASS ○ · WARN ▲ · FAIL ✕")

                fig_rr.update_layout(title=None)
                _style_fig(fig_rr, title=None)
                # Legend title removal: some Plotly.js builds render missing title text as literal "undefined".
                # We keep a legend, but make sure the title field is absent from the serialized figure (see dict-surgery below).
                fig_rr.update_layout(
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="left",
                        x=0,
                        font=dict(size=12),
                    )
                )
                # Clear any grouping metadata that could create group headers.
                for tr in fig_rr.data:
                    try:
                        tr.legendgrouptitle = None
                    except Exception:
                        pass
                    try:
                        tr.legendgroup = None
                    except Exception:
                        pass

                fig_rr.update_layout(margin=dict(l=20, r=20, t=55, b=20))
                fig_rr.update_xaxes(title="Max drawdown (lower is better)", tickformat=".0%")
                if is_currency:
                    fig_rr.update_yaxes(title="Net profit (excluding deposits)", tickprefix="$", separatethousands=True)
                else:
                    fig_rr.update_yaxes(title="Total return (higher is better)", tickformat=".0%")
                
                # Final pass: remove legend title + group title keys entirely from the figure JSON.
                # Some Plotly.js builds render missing legend title text as the literal string "undefined".
                try:
                    _d = fig_rr.to_dict()
                    if "layout" in _d and "legend" in _d["layout"] and isinstance(_d["layout"]["legend"], dict):
                        _d["layout"]["legend"].pop("title", None)
                    for _tr in _d.get("data", []):
                        if isinstance(_tr, dict):
                            _tr.pop("legendgrouptitle", None)
                            _tr.pop("legendgroup", None)
                    fig_rr = go.Figure(_d)
                except Exception:
                    pass

                _plotly(fig_rr)
            else:
                st.caption("Risk/return map unavailable (missing drawdown/return columns).")

            # Robustness distribution (grand robust score)
            score_col = "score.grand_robust" if "score.grand_robust" in df_req.columns else None
            if score_col:
                s = pd.to_numeric(df_req[score_col], errors="coerce").dropna()
                if len(s) > 0:
                    fig_hist = go.Figure(go.Histogram(x=s, nbinsx=35, marker=dict(color=ACCENT_BLUE)))
                    _style_fig(fig_hist, title="Robustness score distribution (after requirements)")
                    fig_hist.update_xaxes(title="Robustness score (higher is better)")
                    fig_hist.update_yaxes(title="Count")
                    _plotly(fig_hist)
    st.markdown("#### Unified candidates table")
    st.caption(f"Candidates shown: **{len(df_show):,}** / Survivors evaluated: **{len(df):,}**")
    if df_show.empty:
        st.info("No candidates under current rules. Relax constraints or run missing tests.")
        st.stop()

    show_cols = [
        "config_id",
        "config.label",
        "grand.verdict",
        "batch.verdict",
        "rsq.verdict",
        "wfq.verdict",
        "score.grand_robust",
        "return_p10",
        "return_p50",
        "dd_p90",
        "uw_days_p90",
        "pct_profitable_windows",
        "pct_windows_traded",
        "twr_p10",
        "twr_p50",
        "robustness_score",
        "performance.twr_total_return",
        "performance.max_drawdown_equity",
        "equity.net_profit_ex_cashflows",
    ]
    show_cols = [c for c in show_cols if c in df_show.columns]

    st.dataframe(df_show.reindex(columns=show_cols), width="stretch", height=520)

    # =========================
    # Top 10 candidate cards (quick inspect)
    # =========================
    st.markdown("#### Top candidates (quick inspect)")
    st.caption("These are the top 10 rows from the table above (same filters + sorting). Click **Inspect** to jump straight to evidence.")
    top_n = int(min(10, len(df_show)))
    _top10 = df_show.head(top_n).copy()

    # Rolling Starts failure threshold comes from the current preference choice.
    # We interpret it as: "a start is a failure if its TWR return is below this tolerance".
    def _rs_failure_threshold_from_answers(ans: Dict[str, int]) -> float:
        try:
            q = next(q for q in rolling_questions() if q.id == "rs_worst_return")
            idx = int(ans.get(q.id, int(q.default_index)))
            idx = max(0, min(idx, len(q.choices) - 1))
            choice = q.choices[idx]
            for c in choice.constraints:
                if str(c.metric_id) == "twr_p10":
                    return float(c.threshold)
        except Exception:
            pass
        # If "Don't filter on this" (or anything weird), default to a simple "below 0%" notion of failure.
        return 0.0

    _rs_fail_thr = _rs_failure_threshold_from_answers(rs_ans)
    _rs_detail = load_rs_detail(run_dir, rs_dir_effective) if rs_dir_effective else None

    def _fmt_pct(x: Any, digits: int = 1) -> str:
        try:
            v = float(x)
            if not math.isfinite(v):
                return "—"
            return f"{v * 100:.{digits}f}%"
        except Exception:
            return "—"

    def _fmt_num(x: Any, digits: int = 2) -> str:
        try:
            v = float(x)
            if not math.isfinite(v):
                return "—"
            return f"{v:,.{digits}f}"
        except Exception:
            return "—"

    def _chip(v: str) -> str:
        v = str(v or "").upper().strip()
        if v == "PASS":
            return "✅ PASS"
        if v == "WARN":
            return "⚠️ WARN"
        if v in {"FAIL", "UNMEASURED"}:
            return "❌ FAIL"
        return v or "—"

    # Percentile for robustness score (within the visible table population).
    _score_col = None
    for c in ["score.grand_robust", "robustness_score"]:
        if c in df_show.columns:
            _score_col = c
            break
    _score_pct = {}
    if _score_col:
        try:
            _s = pd.to_numeric(df_show[_score_col], errors="coerce")
            _r = _s.rank(pct=True, ascending=True)
            _score_pct = {str(cid): float(pct) for cid, pct in zip(df_show["config_id"].astype(str), _r)}
        except Exception:
            _score_pct = {}

    def _top_reason_snippet(row_dict: Dict[str, Any]) -> str:
        # Pick the most severe (critical > warn > info) message across stages, if any.
        sev_rank = {"critical": 0, "warn": 1, "info": 2}
        best = None  # (sev_rank, stage, msg)
        try:
            out_b = evaluate_row_with_questions(row_dict, batch_questions(), batch_ans)
            for v in out_b.violations:
                r = sev_rank.get(str(v.get("severity", "info")).lower(), 9)
                msg = str(v.get("message", "")).strip()
                if msg and (best is None or r < best[0]):
                    best = (r, "Batch", msg)
        except Exception:
            pass

        if rs_sum is not None and not rs_sum.empty:
            try:
                out_r = evaluate_row_with_questions(row_dict, rolling_questions(), rs_ans)
                for v in out_r.violations:
                    r = sev_rank.get(str(v.get("severity", "info")).lower(), 9)
                    msg = str(v.get("message", "")).strip()
                    if msg and (best is None or r < best[0]):
                        best = (r, "RS", msg)
            except Exception:
                pass

        if wf_sum is not None and not wf_sum.empty:
            try:
                out_w = evaluate_row_with_questions(row_dict, walkforward_questions(), wf_ans)
                for v in out_w.violations:
                    r = sev_rank.get(str(v.get("severity", "info")).lower(), 9)
                    msg = str(v.get("message", "")).strip()
                    if msg and (best is None or r < best[0]):
                        best = (r, "WF", msg)
            except Exception:
                pass

        if best is None:
            return "No violations under current preferences."
        msg = best[2]
        if len(msg) > 120:
            msg = msg[:117].rstrip() + "…"
        return f"{best[1]}: {msg}"

    def _rs_fail_rate(config_id: str) -> Optional[float]:
        if _rs_detail is None or _rs_detail.empty or "config_id" not in _rs_detail.columns:
            return None
        d = _rs_detail[_rs_detail["config_id"].astype(str) == str(config_id)].copy()
        if d.empty or "performance.twr_total_return" not in d.columns:
            return None
        vals = pd.to_numeric(d["performance.twr_total_return"], errors="coerce").dropna()
        if vals.empty:
            return None
        return float((vals < float(_rs_fail_thr)).mean())

    cards_cols = st.columns(2, gap="medium")
    for i, (_, r) in enumerate(_top10.iterrows()):
        cid = str(r.get("config_id", "")).strip()
        label = str(r.get("config.label", "")).strip() if "config.label" in _top10.columns else ""
        grand_v = str(r.get("grand.verdict", "")).strip()
        batch_v = str(r.get("batch.verdict", "")).strip()
        rs_v = str(r.get("rsq.verdict", "")).strip()
        wf_v = str(r.get("wfq.verdict", "")).strip()

        rr = r.to_dict()
        reason = _top_reason_snippet(rr)

        # Core stats
        batch_ret = rr.get("performance.twr_total_return", np.nan)
        batch_dd = rr.get("performance.max_drawdown_equity", np.nan)

        rs_p10 = rr.get("twr_p10", np.nan)
        rs_fail = _rs_fail_rate(cid)

        wf_p10 = rr.get("return_p10", np.nan)
        wf_p50 = rr.get("return_p50", np.nan)

        score_pct = _score_pct.get(cid)
        score_line = f"{int(round(score_pct * 100))}th pct" if (score_pct is not None and math.isfinite(score_pct)) else "—"

        # Small "dopamine" signal: a simple grade + a confidence/progress bar.
        _gv = str(grand_v or "").upper().strip()
        if _gv == "PASS":
            if score_pct is not None and score_pct >= 0.90:
                grade = "S"
            elif score_pct is not None and score_pct >= 0.75:
                grade = "A"
            else:
                grade = "A-"
        elif _gv == "WARN":
            grade = "B"
        else:
            grade = "C"

        base = {"PASS": 0.75, "WARN": 0.55, "FAIL": 0.35}.get(_gv, 0.45)
        sp = float(score_pct) if (score_pct is not None and math.isfinite(score_pct)) else 0.50
        # Map percentile (0..1) into a gentle +/- adjustment.
        adj = 0.20 * ((sp - 0.50) * 2.0)  # -0.20 .. +0.20
        confidence = float(max(0.0, min(1.0, base + adj)))

        stage_checks = []
        if batch_v:
            stage_checks.append(str(batch_v).upper())
        if rs_sum is not None and not rs_sum.empty and rs_v:
            stage_checks.append(str(rs_v).upper())
        if wf_sum is not None and not wf_sum.empty and wf_v:
            stage_checks.append(str(wf_v).upper())
        checks_total = len(stage_checks)
        checks_passed = sum(1 for v in stage_checks if v == "PASS")
        checks_ratio = (checks_passed / checks_total) if checks_total else 0.0

        with cards_cols[i % 2]:
            with st.container(border=True):
                hL, hR = st.columns([0.78, 0.22])
                with hL:
                    st.markdown(f"**{label or cid}**")
                    st.caption(f"`{cid}`")
                with hR:
                    st.markdown(f"**#{i + 1}**")
                    st.caption(f"Grade: **{grade}**")

                # Verdict strip (more readable than a long dot-chain)
                tags = [f"**Grand:** {_chip(grand_v)}"]
                if batch_v:
                    tags.append(f"**Batch:** {_chip(batch_v)}")
                if rs_sum is not None and not rs_sum.empty and rs_v:
                    tags.append(f"**RS:** {_chip(rs_v)}")
                if wf_sum is not None and not wf_sum.empty and wf_v:
                    tags.append(f"**WF:** {_chip(wf_v)}")
                st.markdown(" &nbsp;|&nbsp; ".join(tags), unsafe_allow_html=True)

                # Tiny "progress" bars to make scanning satisfying
                pL, pR = st.columns(2)
                with pL:
                    st.caption(f"Checks passed: {checks_passed}/{checks_total}" if checks_total else "Checks passed: —")
                    st.progress(float(checks_ratio))
                with pR:
                    st.caption(f"Overall fit: {int(round(confidence * 100))}/100")
                    st.progress(float(confidence))

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Robustness", score_line)
                with m2:
                    st.metric("Batch return", _fmt_pct(batch_ret))
                with m3:
                    st.metric("Max DD", _fmt_pct(batch_dd))

                n1, n2, n3 = st.columns(3)
                with n1:
                    st.metric("RS p10", _fmt_pct(rs_p10))
                with n2:
                    if rs_fail is None:
                        st.metric("RS fail rate", "—")
                    else:
                        st.metric("RS fail rate", f"{rs_fail * 100:.0f}%")
                        st.caption(f"Fails if start < {_fmt_pct(_rs_fail_thr, digits=0)}")
                with n3:
                    if wf_sum is None or wf_sum.empty:
                        st.metric("WF p10 / p50", "—")
                        st.caption("WF missing")
                    else:
                        st.metric("WF p10 / p50", f"{_fmt_pct(wf_p10)} / {_fmt_pct(wf_p50)}")

                if reason.startswith("No violations"):
                    st.markdown("**Notes:** Clean under current preferences ✅")
                else:
                    st.markdown(f"**Top issue:** {reason}")

                if st.button("Inspect →", key=f"top10.inspect.{cid}", type="primary"):
                    st.session_state["cockpit.pick"] = str(cid)
                    st.session_state["ui.jump_to_evidence"] = True
                    st.session_state["ui.jump_tab"] = "Batch evidence"
                    st.rerun()

    st.divider()

    pick = st.selectbox(
        "Select a strategy to inspect",
        options=df_show["config_id"].astype(str).tolist()[:5000],
        index=0,
        key="cockpit.pick",
    )
    if not pick:
        st.stop()

    row = df2[df2["config_id"].astype(str) == str(pick)].iloc[0].to_dict()

    cfg_map = {r.get("config_id"): r.get("normalized") for r in _load_jsonl(run_dir / "configs_resolved.jsonl")}
    cfg_norm = cfg_map.get(str(pick), {})

    art_dir = top_map.get(str(pick))
    if not (art_dir and art_dir.exists()):
        cache_dir = run_dir / "replay_cache" / str(pick)
        if cache_dir.exists():
            art_dir = cache_dir

    st.divider()

    st.subheader("Evidence")
    st.markdown('<div id="evidence-anchor"></div>', unsafe_allow_html=True)

    _jump_to_evidence = bool(st.session_state.pop("ui.jump_to_evidence", False))
    _jump_tab = str(st.session_state.pop("ui.jump_tab", "Batch evidence")) if _jump_to_evidence else ""
    _tabs_base = ["Autopsy", "Batch evidence", "Rolling Starts evidence", "Walkforward evidence", "Exports"]

    # Keep tab order stable across reruns.
    if "ui.evidence_tab_order" not in st.session_state:
        st.session_state["ui.evidence_tab_order"] = list(_tabs_base)

    if _jump_to_evidence and _jump_tab in _tabs_base:
        st.session_state["ui.evidence_tab_order"] = [_jump_tab] + [t for t in _tabs_base if t != _jump_tab]

    _tabs = st.session_state.get("ui.evidence_tab_order", list(_tabs_base))
    _tab_containers = st.tabs(_tabs)
    _tab = dict(zip(_tabs, _tab_containers))
    if _jump_to_evidence:
        try:
            com.html(
                "<script>try{const el=document.getElementById('evidence-anchor'); if(el){el.scrollIntoView({behavior:'smooth',block:'start'});}}catch(e){};</script>",
                height=0,
            )
        except Exception:
            pass

    with _tab.get("Autopsy", _tab_containers[0]):
        st.markdown("#### Receipts (why the verdict is what it is)")

        def _stage_receipt_block(title: str, q_fn, ans: Dict[str, int]) -> None:
            out = evaluate_row_with_questions(row, q_fn(), ans)
            badge = out.verdict
            st.markdown(f"**{title}: `{badge}`**  —  {out.crits} crit, {out.warns} warn, {out.missing} missing")
            if out.violations:
                vdf = pd.DataFrame(out.violations)
                keep = [c for c in ["severity", "metric", "value", "op", "threshold", "message"] if c in vdf.columns]
                st.dataframe(vdf[keep], width="stretch", height=240)
            elif out.missing_metrics:
                st.caption("No violations, but some metrics were missing for this stage.")
                st.code(", ".join(out.missing_metrics))
            else:
                st.caption("No violations.")

        _stage_receipt_block("Batch", batch_questions, batch_ans)
        if rs_sum is not None and not rs_sum.empty:
            _stage_receipt_block("Rolling Starts", rolling_questions, rs_ans)
        else:
            st.info("Rolling Starts evidence is missing for this run (run it from Build & Run).")
        if wf_sum is not None and not wf_sum.empty:
            _stage_receipt_block("Walkforward", walkforward_questions, wf_ans)
        else:
            st.info("Walkforward evidence is missing for this run (run it from Build & Run).")

        if cfg_norm:
            with st.expander("Config (normalized)", expanded=False):
                st.json(cfg_norm)

    with _tab.get("Batch evidence", _tab_containers[0]):
        st.markdown("#### Backtest build sheet")
        if art_dir and art_dir.exists():
            eq_path = art_dir / "equity_curve.csv"
            cfg_path = art_dir / "config.json"
            met_path = art_dir / "metrics.json"
            tr_path = art_dir / "trades.csv"
            fi_path = art_dir / "fills.csv"


            # ---------------------------
            # Strategy build sheet (SPOT)
            # ---------------------------
            cfg_obj = _read_json(cfg_path)
            met_obj = _read_json(met_path)

            eq_df = _load_csv(eq_path)
            if eq_df is None:
                eq_df = pd.DataFrame()
            tr_df = _load_csv(tr_path)
            if tr_df is None:
                tr_df = pd.DataFrame()

            # Events are optional; if missing we still render the build.
            ev_path = art_dir / "events.csv"
            ev_df = _load_csv(ev_path) if ev_path.exists() else pd.DataFrame()

            rr_sel = row if isinstance(row, dict) else {}
            cid_sel = str(pick)
            label_sel = str(rr_sel.get("label") or rr_sel.get("strategy_label") or rr_sel.get("config_label") or "").strip()
            if not label_sel:
                label_sel = cid_sel

            # --- Config params (spot DCA/swing) ---
            # Prefer replay artifact config.json; fall back to resolved normalized config if missing.
            if not isinstance(cfg_obj, dict) or not cfg_obj:
                cfg_obj = cfg_norm if isinstance(cfg_norm, dict) else {}

            # Support both wrapped config {"strategy_name","side","params":{...}} and older params-only dict.
            if isinstance(cfg_obj, dict) and isinstance(cfg_obj.get("params"), dict):
                params = dict(cfg_obj.get("params") or {})
            else:
                params = dict(cfg_obj) if isinstance(cfg_obj, dict) else {}

            strategy_name = str((cfg_obj.get("strategy_name") if isinstance(cfg_obj, dict) else None) or rr_sel.get("strategy_name") or "strategy").strip()
            side = str((cfg_obj.get("side") if isinstance(cfg_obj, dict) else None) or rr_sel.get("side") or "long").strip().lower()

            def _p(key: str, default: Any) -> Any:
                v = params.get(key, None)
                return default if v is None else v

            # Defaults mirror dca_swing.py behavior.
            deposit_freq = str(_p("deposit_freq", "none") or "none")
            deposit_amt = float(_p("deposit_amount_usd", 0.0) or 0.0)

            buy_freq = str(_p("buy_freq", "weekly") or "weekly")
            buy_amt = float(_p("buy_amount_usd", 0.0) or 0.0)

            buy_filter = str(_p("buy_filter", "none") or "none")
            entry_logic = params.get("entry_logic") if isinstance(params.get("entry_logic"), dict) else None
            n_clauses = len((entry_logic or {}).get("clauses") or []) if entry_logic else 0
            n_regime = len((entry_logic or {}).get("regime") or []) if entry_logic else 0

            max_alloc_pct = float(_p("max_alloc_pct", 1.0) or 1.0)
            sl_pct = float(_p("sl_pct", 0.0) or 0.0)
            trail_pct = float(_p("trail_pct", 0.0) or 0.0)
            max_hold_bars = int(_p("max_hold_bars", 0) or 0)

            tp_pct = float(_p("tp_pct", 0.0) or 0.0)
            tp_sell_fraction = float(_p("tp_sell_fraction", 0.0) or 0.0)

            # --- Core stats from selected row (already in artifacts) ---
            batch_ret = rr_sel.get("performance.twr_total_return", np.nan)
            batch_dd = rr_sel.get("performance.max_drawdown_equity", np.nan)

            rs_p10 = rr_sel.get("twr_p10", np.nan)
            rs_p50 = rr_sel.get("twr_p50", np.nan)
            rs_fail = _rs_fail_rate(cid_sel)

            wf_p10 = rr_sel.get("return_p10", np.nan)
            wf_p50 = rr_sel.get("return_p50", np.nan)
            wf_neg = rr_sel.get("pct_windows_negative", np.nan)

            # Robustness percentile is computed over visible population earlier (same as the cards)
            score_pct = _score_pct.get(cid_sel) if isinstance(_score_pct, dict) else None

            def _clamp01(x: Any) -> float:
                try:
                    v = float(x)
                    if not math.isfinite(v):
                        return 0.0
                    return float(max(0.0, min(1.0, v)))
                except Exception:
                    return 0.0

            # --- Trade stats (derived from trades.csv only) ---
            trade_count = int(len(tr_df)) if tr_df is not None else 0
            pnl_col = _pick_col(tr_df, ["net_pnl", "pnl_after_fees", "pnl", "gross_pnl"]) if trade_count else None
            win_rate = np.nan
            pf = np.nan
            if pnl_col:
                pnl = pd.to_numeric(tr_df[pnl_col], errors="coerce").fillna(0.0).astype(float)
                win_rate = float((pnl > 0).mean()) if len(pnl) else np.nan
                wins = float(pnl[pnl > 0].sum())
                losses = float(pnl[pnl < 0].sum())
                pf = (wins / abs(losses)) if losses < 0 else (float("inf") if wins > 0 else np.nan)

            # Holding time
            med_hold_days = np.nan
            if trade_count and ("entry_dt" in tr_df.columns) and ("exit_dt" in tr_df.columns):
                fmt = "%Y-%m-%d %H:%M:%S%z"
                ent = pd.to_datetime(tr_df["entry_dt"], utc=True, errors="coerce", format=fmt, cache=True)
                ex = pd.to_datetime(tr_df["exit_dt"], utc=True, errors="coerce", format=fmt, cache=True)
                dur = (ex - ent).dt.total_seconds() / 86400.0
                med_hold_days = float(dur.median()) if dur.notna().any() else np.nan

            # Activity (trades / month) using equity curve date span if present
            trades_per_month = np.nan
            if eq_df is not None and not eq_df.empty:
                xcol_tmp = _pick_col(eq_df, ["dt", "timestamp", "time", "date"])
                if xcol_tmp:
                    dts = pd.to_datetime(eq_df[xcol_tmp], utc=True, errors="coerce")
                    dts = dts.dropna()
                    if len(dts) >= 2:
                        span_days = max((dts.max() - dts.min()).days, 1)
                        months = span_days / 30.44
                        trades_per_month = float(trade_count / months) if months > 0 else np.nan
            try:
                if not math.isfinite(float(trades_per_month)):
                    trades_per_month = float(trade_count)
            except Exception:
                trades_per_month = float(trade_count)

            # DCA intensity from events tape if present
            adds_per_entry = np.nan
            entries = 0
            adds = 0
            if ev_df is not None and not ev_df.empty and "event" in ev_df.columns:
                entries = int((ev_df["event"].astype(str) == "ENTRY").sum())
                adds = int((ev_df["event"].astype(str) == "ADD").sum())
                adds_per_entry = float(adds / max(entries, 1))

            # --- Build “traits” (game-style bars; deterministic transforms) ---
            activity_score = _clamp01(float(trades_per_month) / 20.0)  # 20 trades/mo ~ max
            patience_score = _clamp01(float(med_hold_days) / 14.0) if math.isfinite(float(med_hold_days)) else 0.0
            dca_score = _clamp01(float(adds_per_entry) / 3.0) if math.isfinite(float(adds_per_entry)) else 0.0
            toughness_score = _clamp01(1.0 - (float(batch_dd) / 0.25)) if math.isfinite(float(batch_dd)) else 0.0
            consistency_score = _clamp01((float(rs_p10) + 0.10) / 0.25) if math.isfinite(float(rs_p10)) else 0.0
            if rs_fail is not None and math.isfinite(float(rs_fail)):
                consistency_score = _clamp01(consistency_score * (1.0 - float(rs_fail)))
            general_score = _clamp01((float(wf_p10) + 0.10) / 0.25) if math.isfinite(float(wf_p10)) else 0.0

            # “Overall fit” mirrors the Top-10 cards: base by grand verdict + percentile adjustment.
            grand_v = str(rr_sel.get("grand_verdict") or rr_sel.get("verdict") or rr_sel.get("g.verdict") or "").upper().strip()
            base = 0.75 if grand_v == "PASS" else (0.55 if grand_v == "WARN" else 0.35)
            sp = float(score_pct) if (score_pct is not None and math.isfinite(float(score_pct))) else 0.50
            adj = 0.20 * ((sp - 0.50) * 2.0)
            confidence = float(max(0.0, min(1.0, base + adj)))

            # Stage “checks passed”
            batch_v = str(rr_sel.get("batchq.verdict") or rr_sel.get("batch.verdict") or rr_sel.get("batch_verdict") or "").upper().strip()
            rs_v = str(rr_sel.get("rsq.verdict") or rr_sel.get("rs.verdict") or rr_sel.get("rs_verdict") or "").upper().strip()
            wf_v = str(rr_sel.get("wfq.verdict") or rr_sel.get("wf.verdict") or rr_sel.get("wf_verdict") or "").upper().strip()

            stage_checks = []
            if batch_v:
                stage_checks.append(batch_v)
            if rs_sum is not None and not rs_sum.empty and rs_v:
                stage_checks.append(rs_v)
            if wf_sum is not None and not wf_sum.empty and wf_v:
                stage_checks.append(wf_v)
            checks_total = len(stage_checks)
            checks_passed = sum(1 for x in stage_checks if x == "PASS")
            checks_ratio = (checks_passed / checks_total) if checks_total else 0.0

            # Grade (same logic as cards)
            if grand_v == "PASS":
                if score_pct is not None and score_pct >= 0.90:
                    grade = "S"
                elif score_pct is not None and score_pct >= 0.75:
                    grade = "A"
                else:
                    grade = "A-"
            elif grand_v == "WARN":
                grade = "B"
            else:
                grade = "C"

            # Top reason (receipt snippet)
            top_reason = _top_reason_snippet(rr_sel) if "_top_reason_snippet" in globals() or True else ""
            if not top_reason:
                top_reason = "—"

            with st.container(border=True):
                st.caption("Diagnostics derived from saved historical backtest artifacts (spot only, no leverage). Not investment advice.")

                # Header
                hL, hR = st.columns([0.78, 0.22])
                with hL:
                    st.markdown(f"**{label_sel}**")
                    st.caption(f"`{cid_sel}` · `{strategy_name}` · side: `{side}` · market: **spot**")
                with hR:
                    st.markdown(f"**#{int(rr_sel.get('rank', 0) or 0)}**" if rr_sel.get('rank') else "")
                    st.caption(f"Rank band: **{grade}** (relative to this run)")

                # Verdict strip (filters/results flags, not advice)
                tags = [f"**Grand:** {_chip(grand_v)}"]
                if batch_v:
                    tags.append(f"**Batch:** {_chip(batch_v)}")
                if rs_sum is not None and not rs_sum.empty and rs_v:
                    tags.append(f"**RS:** {_chip(rs_v)}")
                if wf_sum is not None and not wf_sum.empty and wf_v:
                    tags.append(f"**WF:** {_chip(wf_v)}")
                st.markdown(" &nbsp;|&nbsp; ".join(tags), unsafe_allow_html=True)

                # Progress bars (gamified, but explicitly relative)
                pL, pR = st.columns(2)
                with pL:
                    st.caption(f"Filters passed: {checks_passed}/{checks_total}" if checks_total else "Filters passed: —")
                    st.progress(float(checks_ratio))
                with pR:
                    st.caption(f"Diagnostics score (relative): {int(round(confidence * 100))}/100")
                    st.progress(float(confidence))

                # Three-panel layout: config | characteristics | diagnostics
                c1, c2, c3 = st.columns([0.38, 0.34, 0.28])

                with c1:
                    st.markdown("**Configuration**")
                    # Render effective schedule/readout (neutral wording; not advice).
                    dep_off = (str(deposit_freq).strip().lower() in {"none", "off", "0", ""} or float(deposit_amt) <= 0.0)
                    buy_off = (str(buy_freq).strip().lower() in {"none", "off", "0", ""} or float(buy_amt) <= 0.0)
                    dep_desc = "`off`" if dep_off else f"`{deposit_freq}` · `${deposit_amt:,.0f}`"
                    buy_desc = "`off`" if buy_off else f"`{buy_freq}` · `${buy_amt:,.0f}`"

                    if entry_logic:
                        if n_clauses or n_regime:
                            entry_desc = f"`logic` (clauses {n_clauses}, regime {n_regime})"
                        else:
                            entry_desc = "`logic`"
                    elif buy_filter and str(buy_filter).lower() != "none":
                        entry_desc = f"`{buy_filter}`"
                    else:
                        entry_desc = f"`scheduled` ({buy_freq})" if not buy_off else "`off`"

                    st.markdown(
                        f"- **Cash additions:** {dep_desc}\n"
                        f"- **Buy schedule:** {buy_desc}\n"
                        f"- **Entry rule:** {entry_desc}\n"
                        f"- **Allocation cap:** `{max_alloc_pct * 100:.0f}%`\n"
                        f"- **Risk controls:** SL `{sl_pct * 100:.1f}%` · Trail `{trail_pct * 100:.1f}%` · Time `{max_hold_bars}` bars\n"
                        f"- **Take profit:** `{tp_pct * 100:.1f}%` · sell `{tp_sell_fraction * 100:.0f}%`"
                    )

                with c2:
                    st.markdown("**Derived characteristics**")
                    tL, tR = st.columns(2)
                    with tL:
                        st.caption(f"Trade frequency · {trades_per_month:.2f}/mo" if math.isfinite(float(trades_per_month)) else "Trade frequency · —")
                        st.progress(float(activity_score))
                        st.caption(f"Median hold · {med_hold_days:.2f} days" if math.isfinite(float(med_hold_days)) else "Median hold · —")
                        st.progress(float(patience_score))
                        st.caption(f"Adds per entry · {adds_per_entry:.2f}" if math.isfinite(float(adds_per_entry)) else "Adds per entry · —")
                        st.progress(float(dca_score))
                    with tR:
                        st.caption(f"Drawdown score · Max DD {_fmt_pct(batch_dd)}" if math.isfinite(float(batch_dd)) else "Drawdown score · —")
                        st.progress(float(toughness_score))
                        st.caption(f"RS stability · p10 {_fmt_pct(rs_p10)}" if math.isfinite(float(rs_p10)) else "RS stability · —")
                        st.progress(float(consistency_score))
                        st.caption(f"WF stability · p10 {_fmt_pct(wf_p10)}" if math.isfinite(float(wf_p10)) else "WF stability · —")
                        st.progress(float(general_score))

                with c3:
                    st.markdown("**Diagnostics**")
                    d1, d2 = st.columns(2)
                    with d1:
                        st.metric("Robustness", f"{int(round(score_pct * 100))}th pct" if (score_pct is not None and math.isfinite(float(score_pct))) else "—")
                        st.metric("Batch return", _fmt_pct(batch_ret))
                    with d2:
                        st.metric("Max DD", _fmt_pct(batch_dd))
                        st.metric("Trades", f"{trade_count}")

                    # RS / WF summaries (text to avoid tall metric stacks)
                    if math.isfinite(float(rs_p10)):
                        fr = "—" if rs_fail is None else f"{rs_fail * 100:.0f}%"
                        st.caption(f"RS: p10 {_fmt_pct(rs_p10)} · p50 {_fmt_pct(rs_p50)} · fail {fr} (thr {_fmt_pct(_rs_fail_thr, digits=0)})")
                    else:
                        st.caption("RS: —")
                    if math.isfinite(float(wf_p10)) or math.isfinite(float(wf_p50)):
                        neg_txt = f"{float(wf_neg) * 100:.0f}% neg" if math.isfinite(float(wf_neg)) else "neg: —"
                        st.caption(f"WF: p10 {_fmt_pct(wf_p10)} · p50 {_fmt_pct(wf_p50)} · {neg_txt}")
                    else:
                        st.caption("WF: missing")

                    # Trade outcome stats
                    wr_txt = f"{win_rate * 100:.0f}%" if math.isfinite(float(win_rate)) else "—"
                    pf_txt = (f"{pf:.2f}" if (pf is not None and math.isfinite(float(pf)) and pf != float('inf')) else ("∞" if pf == float('inf') else "—"))
                    hold_txt = f"{med_hold_days:.2f} d" if math.isfinite(float(med_hold_days)) else "—"
                    st.caption(f"Outcomes: win {wr_txt} · PF {pf_txt} · median hold {hold_txt}")

                # Constraint highlight
                st.markdown(f"**Constraint hit:** {top_reason}")

                with st.expander("Show full configuration", expanded=False):
                    st.json(params if isinstance(params, dict) else cfg_obj)

            st.divider()
            st.markdown("#### Batch replay artifacts")


            # Price + event timeline (receipts on the tape)
            ev_path = art_dir / "events.csv"
            if ev_path.exists():
                st.markdown("##### Price + event timeline (entries/exits/TPs on the tape)")
                ev = _load_csv(ev_path)
                if ev is None or ev.empty:
                    st.info("events.csv exists, but it has 0 rows (likely generated before event logging was added, or the replay generator hit an error).")
                    # Allow regeneration even if cached artifacts exist (needed when new artifact types are added).
                    replay_script = REPO_ROOT / "tools" / "generate_replay_artifacts.py"
                    can_replay = (run_dir / "configs_resolved.jsonl").exists() and replay_script.exists()
                    if st.button("Regenerate replay artifacts (refresh cache)", key=f"replay.regen.empty.{pick}", disabled=(not can_replay)):
                        try:
                            if replay_dir.exists():
                                shutil.rmtree(replay_dir)
                        except Exception:
                            pass
                        cmd = [PY, str(replay_script), "--from-run", str(run_dir), "--config-id", str(pick)]
                        _run_cmd(cmd, cwd=REPO_ROOT, label="Replay: regenerate artifacts")
                        st.rerun()
                else:
                    if "dt" in ev.columns:
                        ev["dt"] = pd.to_datetime(ev["dt"], errors="coerce", utc=True)
                    ev = ev.dropna(subset=["dt"]).sort_values("dt")

                    # Load price series from df_feat (preferred) for the exact run's training tape
                    price = None
                    feat_path = run_dir / "df_feat.parquet"
                    if feat_path.exists():
                        try:
                            # Try fast column-select read first
                            price = pd.read_parquet(feat_path, columns=["dt", "close"])
                        except Exception:
                            try:
                                df_tmp = pd.read_parquet(feat_path)
                                if "dt" in df_tmp.columns and "close" in df_tmp.columns:
                                    price = df_tmp[["dt", "close"]].copy()
                            except Exception:
                                price = None

                    if price is not None and not price.empty and "dt" in price.columns:
                        price["dt"] = pd.to_datetime(price["dt"], errors="coerce", utc=True)
                        price = price.dropna(subset=["dt"]).sort_values("dt")

                        # Focus the view around event range (with buffer)
                        if not ev.empty:
                            lo = ev["dt"].min() - pd.Timedelta(days=7)
                            hi = ev["dt"].max() + pd.Timedelta(days=7)
                            price = price[(price["dt"] >= lo) & (price["dt"] <= hi)]

                        # Downsample for speed (plotly gets sluggish with huge lines)
                        max_points = 3500
                        if len(price) > max_points:
                            idxs = np.linspace(0, len(price) - 1, max_points).astype(int)
                            price = price.iloc[idxs]

                        show_events = st.multiselect(
                            "Show events",
                            ["ENTRY", "ADD", "TP", "EXIT"],
                            default=["ENTRY", "TP", "EXIT"],
                            key=f"ev_show_{pick}",
                        )

                        if go is not None:
                            fig_ev = go.Figure()
                            fig_ev.add_trace(
                                go.Scatter(
                                    x=price["dt"],
                                    y=price["close"],
                                    mode="lines",
                                    name="Close",
                                )
                            )

                            def _add_ev(etype: str, symbol: str, name: str):
                                if etype not in show_events:
                                    return
                                sub = ev[ev.get("event") == etype] if "event" in ev.columns else pd.DataFrame()
                                if sub is None or sub.empty:
                                    return
                                y = pd.to_numeric(sub.get("price"), errors="coerce")
                                text = None
                                if "reason" in sub.columns or "detail" in sub.columns:
                                    if "reason" in sub.columns:
                                        r = sub["reason"].fillna("").astype(str)
                                    else:
                                        r = pd.Series([""] * len(sub), index=sub.index)
                                    if "detail" in sub.columns:
                                        d = sub["detail"].fillna("").astype(str)
                                    else:
                                        d = pd.Series([""] * len(sub), index=sub.index)
                                    text = (r + "\n" + d).str.strip()
                                fig_ev.add_trace(
                                    go.Scatter(
                                        x=sub["dt"],
                                        y=y,
                                        mode="markers",
                                        name=name,
                                        marker=dict(symbol=symbol, size=10),
                                        text=text,
                                        hovertemplate="%{x}<br>%{y}<br>%{text}<extra></extra>" if text is not None else "%{x}<br>%{y}<extra></extra>",
                                    )
                                )

                            _add_ev("ENTRY", "triangle-up", "Entry")
                            _add_ev("ADD", "circle", "Add (DCA)")
                            _add_ev("TP", "diamond", "TP / Partial sell")
                            _add_ev("EXIT", "triangle-down", "Exit")

                            fig_ev.update_layout(
                                height=430,
                                margin=dict(l=10, r=10, t=10, b=10),
                                xaxis_title="Date",
                                yaxis_title="Price",
                                legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="left", x=0, font=dict(size=12)),
                            )
                            _plotly(fig_ev)
                        else:
                            st.info("Plotly is not available; cannot render event timeline chart.")
                    else:
                        st.info("events.csv exists, but df_feat.parquet (dt/close) isn't available for price overlay.")
                st.download_button("Download events.csv", data=ev_path.read_bytes(), file_name=f"{pick}_events.csv")
            else:
                st.caption("No events.csv found for this config yet (replay artifacts need regeneration).")

                # Allow regeneration even if cached artifacts exist (needed when new artifact types are added).
                replay_script = REPO_ROOT / "tools" / "generate_replay_artifacts.py"
                can_replay = (run_dir / "configs_resolved.jsonl").exists() and replay_script.exists()
                if st.button("Regenerate replay artifacts (refresh cache)", key=f"replay.regen.{pick}", disabled=(not can_replay)):
                    try:
                        if replay_dir.exists():
                            shutil.rmtree(replay_dir)
                    except Exception:
                        pass
                    cmd = [PY, str(replay_script), "--from-run", str(run_dir), "--config-id", str(pick)]
                    _run_cmd(cmd, cwd=REPO_ROOT, label="Replay: regenerate artifacts")
                    st.rerun()




            if eq_path.exists():
                eq = _load_csv(eq_path)
                if eq is not None and not eq.empty:
                    if "dt" in eq.columns:
                        eq["dt"] = pd.to_datetime(eq["dt"], errors="coerce", utc=True)

                    # Equity vs contributions (+ optional profit) + drawdown
                    if go is not None and make_subplots is not None and "equity" in eq.columns:
                        try:
                            eq2 = eq.copy()
                            eq2["equity"] = pd.to_numeric(eq2["equity"], errors="coerce")
                            eq2 = eq2.dropna(subset=["equity"])
                            if not eq2.empty:
                                xcol = "dt" if "dt" in eq2.columns else None

                                # Prefer columns precomputed in replay artifacts; fall back for legacy caches.
                                cf = pd.to_numeric(eq2["cashflow"], errors="coerce").fillna(0.0) if "cashflow" in eq2.columns else pd.Series([0.0] * len(eq2), index=eq2.index)
                                deposits = cf.clip(lower=0.0)

                                if "contrib_total" not in eq2.columns:
                                    contrib0 = float(eq2["equity"].iloc[0])
                                    eq2["contrib_total"] = contrib0 + deposits.cumsum()
                                else:
                                    eq2["contrib_total"] = pd.to_numeric(eq2["contrib_total"], errors="coerce")

                                if "profit" not in eq2.columns:
                                    eq2["profit"] = eq2["equity"] - eq2["contrib_total"]
                                else:
                                    eq2["profit"] = pd.to_numeric(eq2["profit"], errors="coerce")

                                peak = eq2["equity"].cummax()
                                eq2["drawdown"] = (eq2["equity"] / peak) - 1.0
                                dd_abs = (eq2["equity"] - peak)

                                # Quick human numbers (so deposits can't gaslight you).
                                last_eq = float(eq2["equity"].iloc[-1])
                                last_contrib = float(eq2["contrib_total"].iloc[-1])
                                last_profit = float(eq2["profit"].iloc[-1])

                                                                # Headline numbers (make the cash-in story unambiguous)
                                initial_cap = float(eq2["contrib_total"].iloc[0]) if "contrib_total" in eq2.columns and len(eq2) else float(eq2["equity"].iloc[0])
                                deposits_only = max(0.0, last_contrib - initial_cap)

                                c1, c2, c3, c4, c5 = st.columns(5)
                                with c1:
                                    st.metric("Initial capital", f"{initial_cap:,.2f}")
                                with c2:
                                    st.metric("Cash in (initial + deposits)", f"{last_contrib:,.2f}")
                                with c3:
                                    st.metric("Deposits only", f"{deposits_only:,.2f}")
                                with c4:
                                    st.metric("Net liquidation value", f"{last_eq:,.2f}")
                                with c5:
                                    st.metric("Profit (equity − cash in)", f"{last_profit:,.2f}")
# Unified hover makes the 3-line story legible.
                                roi = np.where(eq2["contrib_total"].to_numpy() > 0, (eq2["profit"].to_numpy() / eq2["contrib_total"].to_numpy()), np.nan)

                                fig2 = make_subplots(
                                    rows=2,
                                    cols=1,
                                    shared_xaxes=True,
                                    vertical_spacing=0.06,
                                    row_heights=[0.68, 0.32],
                                    subplot_titles=("Equity vs cash in", "Drawdown = drop from running equity peak"),
                                )

                                fig2.add_trace(
                                    go.Scatter(
                                        x=eq2[xcol] if xcol else None,
                                        y=eq2["equity"],
                                        mode="lines",
                                        name="Equity (NLV)",
                                        customdata=np.stack([eq2["contrib_total"].to_numpy(), eq2["profit"].to_numpy(), cf.to_numpy()], axis=1),
                                        hovertemplate="Equity (NLV): %{y:,.2f}<br>Cash in (to date): %{customdata[0]:,.2f}<br>Profit: %{customdata[1]:,.2f}<br>Cashflow this bar: %{customdata[2]:+,.2f}<extra></extra>",
                                        line=dict(width=3, color=ACCENT_BLUE),
                                    ),
                                    row=1,
                                    col=1,
                                )

                                fig2.add_trace(
                                    go.Scatter(
                                        x=eq2[xcol] if xcol else None,
                                        y=eq2["contrib_total"],
                                        mode="lines",
                                        name="Cash in (to date)",
                                        customdata=np.stack([cf.to_numpy()], axis=1),
                                        hovertemplate="Cash in (to date): %{y:,.2f}<br>Cashflow this bar: %{customdata[0]:+,.2f}<extra></extra>",
                                        line=dict(width=2),
                                    ),
                                    row=1,
                                    col=1,
                                )

                                fig2.add_trace(
                                    go.Scatter(
                                        x=eq2[xcol] if xcol else None,
                                        y=eq2["profit"],
                                        mode="lines",
                                        name="Profit (Eq − cash-in)",
                                        customdata=np.stack([roi, cf.to_numpy()], axis=1),
                                        hovertemplate="Profit: %{y:,.2f}<br>ROI on cash in: %{customdata[0]:.2%}<br>Cashflow this bar: %{customdata[1]:+,.2f}<extra></extra>",
                                        line=dict(width=2, dash="dot"),
                                    ),
                                    row=1,
                                    col=1,
                                )

                                fig2.add_trace(
                                    go.Scatter(
                                        x=eq2[xcol] if xcol else None,
                                        y=eq2["drawdown"],
                                        mode="lines",
                                        name="Drawdown",
                                        customdata=np.stack([peak.to_numpy(), dd_abs.to_numpy()], axis=1),
                                        hovertemplate="Drawdown: %{y:.2%}<br>Peak equity: %{customdata[0]:,.2f}<br>Peak→now: %{customdata[1]:,.2f}<extra></extra>",
                                        line=dict(width=2, color=FAIL_COLOR),
                                        fill="tozeroy",
                                        fillcolor="rgba(255,23,68,0.18)",
                                    ),
                                    row=2,
                                    col=1,
                                )

                                fig2.update_yaxes(tickformat=".0f", row=1, col=1)
                                fig2.update_yaxes(tickformat=".0%", row=2, col=1)
                                fig2.update_layout(hovermode="x unified")
                                st.markdown("#### Equity, cash in, profit + drawdown")
                                _style_fig(fig2, title=None)
                                # Title is rendered by Streamlit header; keep Plotly's top margin for a clean legend.
                                fig2.update_layout(
                                    title_text="",
                                    margin=dict(t=85),
                                    legend=dict(
                                        orientation="h",
                                        yanchor="top",
                                        y=1.12,
                                        xanchor="left",
                                        x=0,
                                    ),
                                )
                                st.caption(
                                    "Drawdown = drop from the running equity peak. "
                                    "Cash in = initial capital + deposits (cashflow > 0). "
                                    "Initial capital = cash in at the first bar. "
                                    "Deposits only = cash in − initial capital. "
                                    "Profit = equity − cash in."
                                )
                                _plotly(fig2)
                                with st.expander("Show raw equity curve (table)", expanded=False):
                                    # A simple, no-frills view for sanity checks.
                                    if go is not None:
                                        fig_raw = go.Figure()
                                        fig_raw.add_trace(
                                            go.Scatter(
                                                x=eq2[xcol] if xcol else None,
                                                y=eq2["equity"],
                                                mode="lines",
                                                name="Equity",
                                                customdata=np.stack([cf.to_numpy()], axis=1),
                                                hovertemplate="Equity: %{y:,.2f}<br>Cashflow this bar: %{customdata[0]:+,.2f}<extra></extra>",
                                                line=dict(width=2, color=ACCENT_BLUE),
                                            )
                                        )
                                        fig_raw.add_trace(
                                            go.Scatter(
                                                x=eq2[xcol] if xcol else None,
                                                y=eq2["contrib_total"],
                                                mode="lines",
                                                name="Cash in (to date)",
                                                hovertemplate="Cash in (to date): %{y:,.2f}<extra></extra>",
                                                line=dict(width=1),
                                            )
                                        )
                                        fig_raw.add_trace(
                                            go.Scatter(
                                                x=eq2[xcol] if xcol else None,
                                                y=eq2["profit"],
                                                mode="lines",
                                                name="Profit",
                                                customdata=np.stack([roi], axis=1),
                                                hovertemplate="Profit: %{y:,.2f}<br>ROI on cash in: %{customdata[0]:.2%}<extra></extra>",
                                                line=dict(width=1, dash="dot"),
                                            )
                                        )
                                        fig_raw.update_layout(hovermode="x unified")
                                        _style_fig(fig_raw, title="Raw equity curve (no drawdown)")
                                        _plotly(fig_raw)
                                        # Profit-only view (separate scale) helps when profit is visually squished.
                                        fig_profit = go.Figure()
                                        fig_profit.add_trace(
                                            go.Scatter(
                                                x=eq2[xcol] if xcol else None,
                                                y=eq2["profit"],
                                                mode="lines",
                                                name="Profit",
                                                customdata=np.stack([roi, cf.to_numpy()], axis=1),
                                                hovertemplate="Profit: %{y:,.2f}<br>ROI on cash in: %{customdata[0]:.2%}<br>Cashflow this bar: %{customdata[1]:+,.2f}<extra></extra>",
                                                line=dict(width=2, dash="dot"),
                                            )
                                        )
                                        # Mark deposit/withdraw bars for quick intuition.
                                        if float(np.nanmax(np.abs(cf.to_numpy()))) > 0:
                                            mask = cf != 0
                                            x_ev = (eq2.loc[mask, xcol] if xcol else None)
                                            y_ev = eq2.loc[mask, "profit"]
                                            fig_profit.add_trace(
                                                go.Scatter(
                                                    x=x_ev,
                                                    y=y_ev,
                                                    mode="markers",
                                                    name="Cashflow event",
                                                    customdata=np.stack([cf.loc[mask].to_numpy()], axis=1),
                                                    hovertemplate="Cashflow: %{customdata[0]:+,.2f}<extra></extra>",
                                                )
                                            )
                                        fig_profit.update_layout(hovermode="x unified")
                                        _style_fig(fig_profit, title="Profit only (separate scale)")
                                        _plotly(fig_profit)

                                    # Table is the ultimate audit log.
                                    show_cols = [c for c in ["dt", "equity", "contrib_total", "profit", "drawdown", "cashflow"] if c in eq2.columns]
                                    st.dataframe(eq2[show_cols].tail(500), width="stretch")
                        except Exception:
                            pass
            else:
                st.info("No equity_curve.csv found in artifacts for this config.")


            
            if cfg_path.exists():
                with st.expander("Config (artifact config.json)", expanded=False):
                    st.json(_read_json(cfg_path))
            # Trade outcomes (easy read)
            with st.expander("Trade outcomes (Batch)", expanded=False):
                if tr_path.exists():
                    tr = _load_csv(tr_path)
                    if tr is not None and not tr.empty:
                        st.markdown("##### Trade outcomes (Batch)")
                        pnl_col = _pick_col(tr, ["net_pnl", "gross_pnl", "pnl", "profit"])
                        if pnl_col and pnl_col in tr.columns and go is not None:
                            pnl = pd.to_numeric(tr[pnl_col], errors="coerce").dropna()
                            if len(pnl) > 0:
                                win_rate = float((pnl > 0).mean())
                                avg = float(pnl.mean())
                                med = float(pnl.median())
                                m1, m2, m3 = st.columns(3)
                                with m1:
                                    st.metric("Win rate", f"{win_rate*100:.1f}%")
                                with m2:
                                    st.metric("Avg trade PnL", f"{avg:.2f}")
                                with m3:
                                    st.metric("Median trade PnL", f"{med:.2f}")

                                figp = go.Figure(go.Histogram(x=pnl, nbinsx=40, marker=dict(color=ACCENT_BLUE)))
                                _style_fig(figp, title="Trade PnL distribution")
                                figp.update_xaxes(title=f"{pnl_col}")
                                figp.update_yaxes(title="Count")
                                _plotly(figp)

                        st.caption("Exit reasons are shown above on the price + event timeline (entries/exits/TPs).")

            cdl1, cdl2, cdl3, cdl4 = st.columns(4)
            with cdl1:
                if met_path.exists():
                    st.download_button("Download metrics.json", data=met_path.read_bytes(), file_name=f"{pick}_metrics.json")
            with cdl2:
                if tr_path.exists():
                    st.download_button("Download trades.csv", data=tr_path.read_bytes(), file_name=f"{pick}_trades.csv")
            with cdl3:
                if fi_path.exists():
                    st.download_button("Download fills.csv", data=fi_path.read_bytes(), file_name=f"{pick}_fills.csv")
            with cdl4:
                if eq_path.exists():
                    st.download_button("Download equity_curve.csv", data=eq_path.read_bytes(), file_name=f"{pick}_equity_curve.csv")
        else:
            st.info("No saved artifacts for this config yet.")
            replay_script = REPO_ROOT / "tools" / "generate_replay_artifacts.py"
            can_replay = (run_dir / "configs_resolved.jsonl").exists() and replay_script.exists()
            if st.button("Generate replay artifacts (cached)", key="replay.gen", disabled=(not can_replay)):
                cmd = [PY, str(replay_script), "--from-run", str(run_dir), "--config-id", str(pick)]
                _run_cmd(cmd, cwd=REPO_ROOT, label="Replay: generate artifacts")
                st.rerun()

    with _tab.get("Rolling Starts evidence", _tab_containers[0]):
        st.markdown("#### Rolling Starts detail")
        if rs_dir_effective and (rs_dir_effective / "rolling_starts_detail.csv").exists():
            rs_det = load_rs_detail(run_dir, rs_dir_effective)
            if rs_det is not None and not rs_det.empty and "config_id" in rs_det.columns:
                d = rs_det[rs_det["config_id"].astype(str) == str(pick)].copy()
                if d.empty:
                    st.info("No Rolling Starts detail rows for this config.")
                else:
                    if "start_dt" in d.columns:
                        d["start_dt"] = pd.to_datetime(d.get("start_dt"), errors="coerce", utc=True)
                    if px is not None and "performance.twr_total_return" in d.columns:
                        fig = px.histogram(
                            d,
                            x="performance.twr_total_return",
                            nbins=40,
                            title="Rolling-start TWR distribution (detail)",
                        )
                        _style_fig(fig, title="Rolling-start TWR distribution (detail)")
                        _plotly(fig)


                    # Timeline view: start date → outcome (helps spot "bad eras")
                    if go is not None and "start_dt" in d.columns and "performance.twr_total_return" in d.columns:
                        try:
                            dd = d.copy()
                            dd["performance.twr_total_return"] = pd.to_numeric(dd["performance.twr_total_return"], errors="coerce")
                            dd = dd.dropna(subset=["performance.twr_total_return", "start_dt"])
                            if not dd.empty:
                                figt = go.Figure()
                                figt.add_trace(
                                    go.Scatter(
                                        x=dd["start_dt"],
                                        y=dd["performance.twr_total_return"],
                                        mode="markers",
                                        marker=dict(size=7, color=ACCENT_BLUE, opacity=0.8),
                                        name="Rolling starts",
                                        hovertemplate="%{x|%Y-%m-%d}<br>return=%{y:.2%}<extra></extra>",
                                    )
                                )
                                _style_fig(figt, title="Rolling Starts timeline (each dot = a different start date)")
                                figt.update_yaxes(tickformat=".0%", title="Total return")
                                figt.update_xaxes(title="Start date")
                                _plotly(figt)
                        except Exception:
                            pass
                    st.download_button(
                        "Download rolling_starts_detail.csv (full)",
                        data=(rs_dir_effective / "rolling_starts_detail.csv").read_bytes(),
                        file_name=f"{selected_run_name}_rolling_starts_detail.csv",
                    )
        else:
            st.info("Rolling Starts evidence not available for this run (run it from Build & Run).")

    with _tab.get("Walkforward evidence", _tab_containers[0]):
        st.markdown("#### Walkforward detail")
        if wf_dir_effective and (wf_dir_effective / "wf_results.csv").exists():
            wf_rows = load_wf_results(wf_dir_effective)
            if wf_rows is not None and not wf_rows.empty and "config_id" in wf_rows.columns:
                pick_eff = str(pick).strip()

                # Fallback: if user picked something non-canonical (e.g., a legacy integer id),
                # try to map it to the real config_id using the current candidates table.
                if ("config.id" in df2.columns) and (pick_eff.isdigit() or (pick_eff and not pick_eff.startswith("cfg_"))):
                    # 1) If it's a config line number, map line_no -> config_id
                    if pick_eff.isdigit() and ("config.line_no" in df2.columns):
                        try:
                            li = int(pick_eff)
                            m = df2[df2["config.line_no"].astype(int) == li]
                            if len(m) == 1:
                                pick_eff = str(m["config_id"].iloc[0]).strip()
                        except Exception:
                            pass

                    # 2) If it's an index-like integer, treat it as 1-based row id into df_show (common confusion).
                    if pick_eff.isdigit():
                        try:
                            i = int(pick_eff)
                            if 1 <= i <= len(df_show):
                                pick_eff = str(df_show["config_id"].astype(str).iloc[i - 1]).strip()
                        except Exception:
                            pass

                d = wf_rows[wf_rows["config_id"].astype(str).str.strip() == str(pick_eff)].copy()
                if d.empty:
                    st.info("No Walkforward detail rows for this config.")
                else:


                    if go is not None and "window_return" in d.columns:
                        try:
                            dd = d.copy()
                            dd["window_return"] = pd.to_numeric(dd["window_return"], errors="coerce")
                            dd = dd.dropna(subset=["window_return"])
                            if not dd.empty:
                                x = dd["window_idx"] if "window_idx" in dd.columns else list(range(len(dd)))
                                cols = [PASS_COLOR if float(v) >= 0.0 else FAIL_COLOR for v in dd["window_return"].tolist()]
                                fig = go.Figure(go.Bar(x=x, y=dd["window_return"], marker_color=cols))
                                _style_fig(fig, title="Walkforward window returns (detail)")
                                fig.update_yaxes(tickformat=".0%", title="Window return")
                                fig.update_xaxes(title="Window")
                                _plotly(fig)
                        except Exception:
                            pass
                    elif px is not None and "window_return" in d.columns:
                        fig = px.line(d, x="window_idx", y="window_return", title="Walkforward window returns (detail)")
                        _style_fig(fig, title="Walkforward window returns (detail)")
                        _plotly(fig)
                    st.download_button(
                        "Download wf_results.csv (full)",
                        data=(wf_dir_effective / "wf_results.csv").read_bytes(),
                        file_name=f"{selected_run_name}_wf_results.csv",
                    )
        else:
            st.info("Walkforward evidence not available for this run (run it from Build & Run).")

    with _tab.get("Exports", _tab_containers[-1]):
        st.markdown("#### Exports")

        st.download_button(
            "Download candidates (CSV)",
            data=df2.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_run_name}_candidates.csv",
        )

        st.download_button(
            "Download manifest.json",
            data=json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name=f"{selected_run_name}_manifest.json",
            key="exports.manifest",
        )

        st.divider()
        st.markdown("#### Strategy pack (.zip)")
        st.caption("Portable bundle for one strategy: manifest + config + filtered Batch/RS/WF evidence, with verifiable hashes (Pack v2).")

        copt1, copt2 = st.columns(2)
        with copt1:
            include_replay = st.checkbox("Include replay/top artifacts", value=True, key="pack.include_replay")
        with copt2:
            include_dataset = st.checkbox("Include dataset file (usually off)", value=False, key="pack.include_dataset")

        pack_key = f"pack.bytes.{selected_run_name}.{pick}.v2"
        if st.button("Build strategy pack", key="pack.build", type="primary"):
            try:
                pack_bytes = _build_strategy_pack_zip(
                    run_dir=run_dir,
                    run_name=str(selected_run_name),
                    config_id=str(pick),
                    manifest=manifest,
                    candidate_row=row,
                    cfg_norm=cfg_norm,
                    rs_dir=rs_dir_effective if isinstance(rs_dir_effective, Path) else None,
                    wf_dir=wf_dir_effective if isinstance(wf_dir_effective, Path) else None,
                    top_art_dir=art_dir if (art_dir and art_dir.exists()) else None,
                    include_replay=bool(include_replay),
                    include_dataset=bool(include_dataset),
                )
                st.session_state[pack_key] = pack_bytes
                st.success("Strategy pack is ready.")
            except Exception as e:
                st.error(f"Failed to build strategy pack: {e}")

        if pack_key in st.session_state:
            st.download_button(
                "Download strategy pack (.zip)",
                data=st.session_state[pack_key],
                file_name=f"{selected_run_name}_strategy_pack_v2_{str(pick)[:10]}.zip",
                key="pack.download",
            )

        st.divider()
        st.markdown("#### Verify a strategy pack (.zip)")
        up = st.file_uploader("Upload a strategy pack to verify", type=["zip"], key="pack.verify.upload")
        if up is not None:
            try:
                z = zipfile.ZipFile(io.BytesIO(up.getvalue()), "r")
                names = set(z.namelist())
                missing = [p for p in ["manifest.json", "meta/pack_index.json", "README.md"] if p not in names]
                if missing:
                    st.warning("Missing required files: " + ", ".join(missing))

                pack_index = {}
                try:
                    pack_index = json.loads(z.read("meta/pack_index.json").decode("utf-8"))
                except Exception:
                    pack_index = {}

                ok = True
                mismatches = []
                files = (pack_index.get("files") or {}) if isinstance(pack_index, dict) else {}
                if files:
                    for arc, rec in files.items():
                        try:
                            b = z.read(arc)
                            dig = hashlib.sha256(b).hexdigest()
                            exp = str((rec or {}).get("digest") or "")
                            if exp and dig != exp:
                                ok = False
                                mismatches.append((arc, exp, dig))
                        except KeyError:
                            ok = False
                            mismatches.append((arc, "missing", "missing"))
                        except Exception:
                            pass
                else:
                    st.info("No pack_index hashes found; cannot fully verify integrity.")

                if ok and files:
                    st.success("Pack integrity verified (hashes match).")
                elif mismatches:
                    st.error("Pack integrity issues found.")
                    for arc, exp, got in mismatches[:20]:
                        st.write(f"- {arc}: expected {exp[:10]}… got {got[:10]}…")

                # Dataset verification (optional)
                try:
                    man = json.loads(z.read("manifest.json").decode("utf-8"))
                except Exception:
                    man = {}
                ds = (man.get("dataset") or {}) if isinstance(man, dict) else {}
                fp = (ds.get("fingerprint") or {}) if isinstance(ds, dict) else {}
                exp_digest = str(fp.get("digest") or "")

                if exp_digest:
                    st.caption("Optional: verify a local dataset file against this pack's dataset fingerprint.")
                    ds_path = st.text_input("Local dataset path", value="", key="pack.verify.dataset_path")
                    if st.button("Verify dataset fingerprint", key="pack.verify.dataset_btn") and ds_path:
                        p = Path(ds_path).expanduser()
                        if not p.exists():
                            st.error("Dataset file not found at that path.")
                        else:
                            cur = _fingerprint_file(p)
                            got = str(cur.get("digest") or "")
                            if got == exp_digest:
                                st.success("Dataset fingerprint matches (comparable).")
                            else:
                                st.error("Dataset fingerprint does NOT match.")
            except Exception as e:
                st.error(f"Could not verify pack: {e}")