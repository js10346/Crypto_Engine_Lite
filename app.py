
from __future__ import annotations

import json
import os
import math
import re
import subprocess
import sys
import time
import threading
import queue
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    px = None
    go = None

# Optional (nice formatting + metric labels)
try:
    from lab.metrics import METRICS
except Exception:  # pragma: no cover
    METRICS = {}

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
        st.info("Run monitor will appear here once progress telemetry is available.")
        return

    paths: List[Path] = []
    if progress_path.is_dir():
        paths = sorted(progress_path.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not paths:
            st.info("Run monitor will appear here once progress telemetry is available.")
            return
    else:
        if not progress_path.exists():
            st.info("Run monitor will appear here once progress telemetry is available.")
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
        st.info("Run monitor will appear here once progress telemetry is available.")
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
    phase = str(last.get("phase", ""))
    done = last.get("i", last.get("done", last.get("n_done", 0)))
    total = last.get("n", last.get("total", last.get("n_total", 0)))

    # Header metrics
    cols = st.columns(4)
    with cols[0]:
        st.metric("Stage", (f"{stage}:{phase}" if phase else stage) or "(unknown)")
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



def _run_cmd(
    cmd: List[str],
    *,
    cwd: Path,
    label: str,
    progress_path: Optional[Path] = None,
    refresh_hz: float = 4.0,
) -> None:
    """Run a command and stream output + telemetry into the UI."""
    if not cmd:
        raise ValueError("Empty command")

    # Make "python" consistent across platforms
    if str(cmd[0]).lower() in {"python", "py", "py.exe", "python3"}:
        cmd = [PY, *cmd[1:]]

    with st.expander(label, expanded=True):
        st.code(" ".join(cmd), language="bash")

        mon_ph = st.empty()
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

        lines = deque(maxlen=240)
        sleep_s = max(0.05, 1.0 / float(max(1.0, refresh_hz)))

        while p.poll() is None:
            # Drain output queue
            for _ in range(200):
                try:
                    lines.append(q.get_nowait())
                except Exception:
                    break

            with mon_ph.container():
                _render_run_monitor(progress_path)

            if lines:
                log_ph.code("".join(list(lines)[-120:]), language="text")

            time.sleep(sleep_s)

        # Final drain
        for _ in range(2000):
            try:
                lines.append(q.get_nowait())
            except Exception:
                break

        dt = time.time() - t0
        if lines:
            log_ph.code("".join(lines), language="text")

        rc = int(p.returncode or 0)
        if rc != 0:
            raise RuntimeError(f"Command failed (code={rc}) after {dt:.1f}s")


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
    if df is None or df.empty:
        return df
    out = df.copy()
    if "config_id" not in out.columns:
        if "config.id" in out.columns:
            out["config_id"] = out["config.id"].astype(str)
        elif "config_id" in out.columns:
            out["config_id"] = out["config_id"].astype(str)
    else:
        out["config_id"] = out["config_id"].astype(str)
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
        df["config_id"] = df["config_id"].astype(str)
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
        df["config_id"] = df["config_id"].astype(str)
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
        df["config_id"] = df["config_id"].astype(str)
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
        df["config_id"] = df["config_id"].astype(str)
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

def build_dca_baseline_params() -> Dict[str, Any]:
    st.subheader("Baseline plan (DCA/Swing)")

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

    with colB:
        buy_filter = st.selectbox(
            "Only buy when…",
            options=[
                "none",
                "below_ema",
                "rsi_below",
            ],
            index=0,
            key="new.buy_filter",
        )
        ema_len = 200
        rsi_thr = 40.0
        if buy_filter == "below_ema":
            ema_len = int(
                st.number_input("EMA length", min_value=5, max_value=600, value=200, step=5, key="new.ema_len")
            )
        if buy_filter == "rsi_below":
            rsi_thr = float(
                st.number_input("RSI threshold (buy when RSI < threshold)", min_value=5.0, max_value=80.0, value=40.0, step=1.0, key="new.rsi_thr")
            )

        max_alloc_pct = float(
            st.slider("Max allocation (fraction of equity)", min_value=0.05, max_value=1.00, value=1.00, step=0.05, key="new.max_alloc_pct")
        )

        with st.expander("Advanced risk controls", expanded=False):
            sl_pct = float(st.slider("Stop loss % (0 disables)", 0.0, 0.80, 0.0, 0.01, key="new.sl_pct"))
            tp_pct = float(st.slider("Take profit % (0 disables)", 0.0, 2.0, 0.0, 0.01, key="new.tp_pct"))
            tp_sell_fraction = float(st.slider("TP sell fraction", 0.0, 1.0, 1.0, 0.05, key="new.tp_sell_frac"))
            reserve_frac = float(st.slider("Reserve fraction of TP proceeds", 0.0, 1.0, 0.0, 0.05, key="new.reserve_frac"))

    params = {
        "deposit_freq": deposit_freq,
        "deposit_amount_usd": float(deposit_amount),
        "buy_freq": buy_freq,
        "buy_amount_usd": float(buy_amount),
        "buy_filter": buy_filter,
        "ema_len": int(ema_len),
        "rsi_thr": float(rsi_thr),
        "max_alloc_pct": float(max_alloc_pct),
        "sl_pct": float(st.session_state.get("new.sl_pct", 0.0)),
        "tp_pct": float(st.session_state.get("new.tp_pct", 0.0)),
        "tp_sell_fraction": float(st.session_state.get("new.tp_sell_frac", 1.0)),
        "reserve_frac_of_proceeds": float(st.session_state.get("new.reserve_frac", 0.0)),
    }

    # Plan summary (plain English)
    parts = []
    if params["deposit_freq"] != "none" and params["deposit_amount_usd"] > 0:
        parts.append(f"Deposit ${_fmt_money(params['deposit_amount_usd'])} {params['deposit_freq']}.")
    parts.append(f"Buy ${_fmt_money(params['buy_amount_usd'])} {params['buy_freq']}.")
    if buy_filter == "below_ema":
        parts.append(f"Only buy if price < EMA({params['ema_len']}).")
    elif buy_filter == "rsi_below":
        parts.append(f"Only buy if RSI < {params['rsi_thr']:.0f}.")
    parts.append(f"Never allocate more than {_fmt_pct(params['max_alloc_pct'], digits=0)} of equity.")
    st.info(" ".join(parts))

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

    st.divider()
    st.header("Workflow")

    # Stage nav (simple stepper)
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

    stage_map = {key: label for (label, key) in STAGES}
    stage_pick = st.radio("Stage", options=stage_keys, format_func=lambda k: stage_map.get(k, k), key="ui.stage")

    st.divider()
    st.caption("Tip: start with Batch. Only run RS/WF once you have survivors.")

# =============================================================================
# New run wizard (when "(new run)" is selected)
# =============================================================================

if open_existing == "(new run)":
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
                                st.plotly_chart(fig, use_container_width=True)
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

        vary_groups = ["deposits", "buys", "filter", "alloc", "risk"]
        vary = vary_groups
        if mode.startswith("Stress test"):
            st.write("What should be allowed to vary?")
            cols = st.columns(len(vary_groups))
            picks: List[str] = []
            for i, g in enumerate(vary_groups):
                with cols[i]:
                    if st.checkbox(g, value=True, key=f"new.vary.{g}"):
                        picks.append(g)
            vary = picks or ["deposits", "buys"]  # never allow empty
        else:
            # random mode ignores vary/width/base
            vary = vary_groups

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

        do_run = st.button("Run batch stress test", type="primary")

        if do_run:
            try:
                t0 = time.time()
                tmp_run_dir = tmp_dir / f"run_{_now_slug()}"
                tmp_run_dir.mkdir(parents=True, exist_ok=True)

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

                _run_cmd(grid_cmd, cwd=REPO_ROOT, label="1) Generate variants grid")
                _ensure_grid_has_baseline(grid_path, base_path, total_n=int(st.session_state["new.grid_n"]))

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
                _run_cmd(
                    batch_cmd,
                    cwd=REPO_ROOT,
                    label="2) Batch sweep + rerun",
                    progress_path=batch_progress.parent,
                )

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

                _run_cmd(post_cmd, cwd=REPO_ROOT, label="3) Postprocess (rank + top_ids)")

                st.success(f"Done in {time.time()-t0:.1f}s. Run saved to: {run_dir.name}")

                # Switch to opening this run
                st.session_state["selected_run"] = run_dir.name
                st.session_state["ui.open_run_next"] = run_dir.name  # set on next rerun before widget instantiates

                # Reset wizard
                st.session_state["new.step"] = 0

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

        st.plotly_chart(fig, use_container_width=True)

        # Quick sanity: list frontier points (helps confirm it's "real" and not plotting artifacts)
        with st.expander("Pareto frontier points", expanded=False):
            if frontier.empty:
                st.caption("No frontier points available (check filters).")
            else:
                show_cols = [c for c in ["config_id", "_label", "_verdict", "_trades", "_dd", "_profit"] if c in frontier.columns]
                st.dataframe(frontier[show_cols].sort_values("_dd", ascending=True), use_container_width=True)
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
    heat["config_id"] = heat_base["config_id"].astype(str)
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

    st.dataframe(sty, use_container_width=True, height=420)

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
                        st.plotly_chart(fig, use_container_width=True)

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
                            st.plotly_chart(fig2, use_container_width=True)
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
                        st.plotly_chart(fig, use_container_width=True)

                    with tab_exp:
                        if "exposure" in plot_df.columns and plot_df["exposure"].notna().any():
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=plot_df["dt"], y=plot_df["exposure"], mode="lines", name="Exposure (pos_value / equity)"))
                            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320, legend=dict(orientation="h"))
                            st.plotly_chart(fig, use_container_width=True)
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
                                st.dataframe(tdf.sort_values("net_pnl", ascending=False).head(5)[show_cols or ["net_pnl"]], use_container_width=True, height=220)
                            with right:
                                st.caption("Worst (by net_pnl)")
                                st.dataframe(tdf.sort_values("net_pnl", ascending=True).head(5)[show_cols or ["net_pnl"]], use_container_width=True, height=220)

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
                st.dataframe(df_show[cols], use_container_width=True, height=520)

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
                str(rs_out_dir),
                "--ids",
                str(ids_file),
                "--top-n",
                str(len(survivors_ids)),
                "--start-step",
                str(start_step),
                "--min-bars",
                str(min_bars_effective),
                "--seed",
                "1",
                "--starting-equity",
                str(float(meta.get("starting_equity", 1000.0) or 1000.0)),
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
        st.plotly_chart(fig, use_container_width=True)

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
                            st.plotly_chart(fig_r, use_container_width=True)

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
                                        use_container_width=True,
                                        height=210,
                                    )
                                with cc2:
                                    st.write("**Best starts (by return)**")
                                    st.dataframe(
                                        best[[c for c in ["start_dt", "start_i", "performance.twr_total_return", "performance.max_drawdown_equity", "uw_max_days"] if c in best.columns]],
                                        use_container_width=True,
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
                            st.plotly_chart(fig_d, use_container_width=True)

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
                            st.plotly_chart(fig_u, use_container_width=True)

                    
                    with tabs[3]:
                        # Distribution views help you see "how often does it hurt?"
                        cols_dist = st.columns(3)

                        if "performance.twr_total_return" in g.columns:
                            g["_twr_pct"] = pd.to_numeric(g["performance.twr_total_return"], errors="coerce") * 100.0
                            fig_hd = px.histogram(g.dropna(subset=["_twr_pct"]), x="_twr_pct", nbins=30, title="Return distribution (rolling starts)")
                            for qname, dash in [("twr_p10", "dot"), ("twr_p50", "dash"), ("twr_p90", "dot")]:
                                if qname in row.columns and not pd.isna(r0.get(qname)):
                                    fig_hd.add_vline(x=float(r0.get(qname)) * 100.0, line_dash=dash)
                            cols_dist[0].plotly_chart(fig_hd, use_container_width=True)
                        else:
                            cols_dist[0].info("No return column in detail.")

                        if "performance.max_drawdown_equity" in g.columns:
                            g["_dd_pct"] = pd.to_numeric(g["performance.max_drawdown_equity"], errors="coerce") * 100.0
                            fig_dd = px.histogram(g.dropna(subset=["_dd_pct"]), x="_dd_pct", nbins=30, title="Drawdown distribution (rolling starts)")
                            for qname, dash in [("dd_p50", "dash"), ("dd_p90", "dot")]:
                                if qname in row.columns and not pd.isna(r0.get(qname)):
                                    fig_dd.add_vline(x=float(r0.get(qname)) * 100.0, line_dash=dash)
                            cols_dist[1].plotly_chart(fig_dd, use_container_width=True)
                        else:
                            cols_dist[1].info("No drawdown column in detail.")

                        if "uw_max_days" in g.columns:
                            fig_uw = px.histogram(g.dropna(subset=["uw_max_days"]), x="uw_max_days", nbins=30, title="Underwater days distribution (rolling starts)")
                            for qname, dash in [("uw_p50_days", "dash"), ("uw_p90_days", "dot")]:
                                if qname in row.columns and not pd.isna(r0.get(qname)):
                                    fig_uw.add_vline(x=float(r0.get(qname)), line_dash=dash)
                            cols_dist[2].plotly_chart(fig_uw, use_container_width=True)
                        else:
                            cols_dist[2].info("No underwater column in detail.")

                        st.caption("Dashed line = median. Dotted lines = p10/p90 (or p90 for drawdown/underwater). Tight distributions are what 'robust' looks like.")
                    with tabs[4]:
                        st.dataframe(g, use_container_width=True, height=420)

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
            st.dataframe(df_show[cols], use_container_width=True, height=520)

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
    st.dataframe(df_show[cols], use_container_width=True, height=520)

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
            st.plotly_chart(fig, use_container_width=True)
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
                st.plotly_chart(fig2, use_container_width=True)
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
                st.plotly_chart(fig_uw, use_container_width=True)
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
                st.plotly_chart(fig3, use_container_width=True)
                st.caption("Histogram of window returns. p10 is the 'worst-typical' anchor; p50 is typical; p90 is best-typical.")

            # Window leaderboard (failure modes)
            if "window_return" in wsub.columns:
                st.write("**Window leaderboard (failure modes)**")
                show_cols = [c for c in ["window_idx", "window_start_dt", "window_end_dt", "window_return", "window_max_drawdown", "window_underwater_days", "trades_closed", "flags"] if c in wsub.columns]

                t1, t2, t3 = st.tabs(["Worst return", "Worst drawdown", "Longest underwater"])
                with t1:
                    st.dataframe(
                        wsub.sort_values("window_return", ascending=True)[show_cols].head(10),
                        use_container_width=True,
                        height=260,
                    )
                with t2:
                    if "window_max_drawdown" in wsub.columns:
                        st.dataframe(
                            wsub.sort_values("window_max_drawdown", ascending=False)[show_cols].head(10),
                            use_container_width=True,
                            height=260,
                        )
                    else:
                        st.info("No drawdown column for this walkforward run.")
                with t3:
                    if "window_underwater_days" in wsub.columns:
                        st.dataframe(
                            wsub.sort_values("window_underwater_days", ascending=False)[show_cols].head(10),
                            use_container_width=True,
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
                    st.plotly_chart(fig4, use_container_width=True)

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
    st.write("### D) Grand verdict (Batch + RS + WF)")
    st.caption(
        "This is the final filter + ranking stage. You pick your pain limits first (target profile), "
        "then the lab finds the strategies that survive *all* stress tests."
    )

    with st.expander("How to read Grand Verdict", expanded=False):
        st.markdown(
            """
- **PASS / WARN / FAIL** are driven by the question sets below.
- **PASS** means the strategy stays within your limits.
- **WARN** means it violates *soft* constraints (it might be OK, but it’s suspicious).
- **FAIL** means it violates *hard* constraints.
- If Rolling Starts or Walkforward are **UNMEASURED**, you can either ignore them or require them.

The ranking is intentionally **worst‑case aware**: it prefers strategies with good *p10* outcomes and tolerable *p90* drawdowns.
            """.strip()
        )

    # Load latest RS/WF if present
    rs_dir_effective = rs_latest
    wf_dir_effective = wf_latest

    rs_sum = load_rs_summary(run_dir, rs_dir_effective) if rs_dir_effective else None
    wf_sum = load_wf_summary(wf_dir_effective) if wf_dir_effective else None

    df = survivors.copy()
    df = _ensure_config_id(df)

    # --- Target profile presets ---
    preset = st.selectbox(
        "Target profile preset",
        options=["None", "Balanced", "Conservative", "Aggressive"],
        index=1,
        key="grand.profile_preset",
        help="Sets default choices for the filters below. You can still tweak anything.",
    )
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Apply preset", key="grand.apply_preset_btn"):
            _apply_grand_preset(str(preset))
            st.rerun()
    with c2:
        st.caption("Presets just set defaults for the radios below. They don’t change your data, only the filters.")

    # Stage A verdict
    with st.expander("Batch questions (used in grand verdict)", expanded=False):
        batch_ans = _question_ui(batch_questions(), key_prefix="q.grand.batch")
    df = apply_stage_eval(df, stage_key="batch", questions=batch_questions(), answers=batch_ans)

    # Stage B verdict (if measured)
    if rs_sum is not None and not rs_sum.empty:
        df = merge_stage(df, rs_sum, on="config_id", suffix="rs")
        with st.expander("Rolling-start questions (used in grand verdict)", expanded=False):
            rs_ans = _question_ui(rolling_questions(), key_prefix="q.grand.rs")
        df = apply_stage_eval(df, stage_key="rsq", questions=rolling_questions(), answers=rs_ans)
    else:
        df["rs.measured"] = False
        df["rsq.verdict"] = "UNMEASURED"

    # Stage C verdict (if measured)
    if wf_sum is not None and not wf_sum.empty:
        df = merge_stage(df, wf_sum, on="config_id", suffix="wf")
        with st.expander("Walkforward questions (used in grand verdict)", expanded=False):
            wf_ans = _question_ui(walkforward_questions(), key_prefix="q.grand.wf")
        df = apply_stage_eval(df, stage_key="wfq", questions=walkforward_questions(), answers=wf_ans)
    else:
        df["wf.measured"] = False
        df["wfq.verdict"] = "UNMEASURED"

    st.divider()
    st.subheader("Grand filter rules")

    col1, col2, col3 = st.columns(3)
    with col1:
        req_batch = st.selectbox("Require Batch", options=["PASS only", "PASS or WARN", "Ignore"], index=1, key="grand.req_batch")
    with col2:
        req_rs = st.selectbox("Require Rolling Starts", options=["PASS only", "PASS or WARN", "Ignore"], index=1, key="grand.req_rs")
    with col3:
        req_wf = st.selectbox("Require Walkforward", options=["PASS only", "PASS or WARN", "Ignore"], index=1, key="grand.req_wf")

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

        # Combine into a single grand verdict (for readability)
        if "FAIL" in stage_vs or "UNMEASURED" in stage_vs:
            gv = "FAIL" if "UNMEASURED" not in stage_vs else "UNMEASURED"
        elif "WARN" in stage_vs:
            gv = "WARN"
        else:
            gv = "PASS"
        grand_verdicts.append(gv)

    df["grand.verdict"] = grand_verdicts
    df2 = df[pd.Series(keep_mask, index=df.index)].copy()

    st.success(f"Grand survivors: {len(df2)}/{len(df)}")

    # --- Scoring / ranking ---
    st.subheader("Ranking")

    # Compute a worst-case aware score (visible + exportable)
    df2["score.grand_robust"] = [_grand_score_row(r) for r in df2.to_dict(orient="records")]
    df2["score.grand_robust"] = pd.to_numeric(df2["score.grand_robust"], errors="coerce")

    with st.expander("What is the 'Grand robust score'?", expanded=False):
        st.markdown(
            """
It’s a deliberately boring scoring function:

- rewards **p10 returns** (worst‑typical outcomes)  
- penalizes **p90 drawdowns** (bad‑typical pain)  
- lightly penalizes **underwater time** (in years)

Walkforward matters most, Rolling Starts second, Batch third.

It’s not a magic number — it’s a ranking hint. The receipts (charts + artifacts) still win.
            """.strip()
        )

    sort_opts = []
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

    sort_by = st.selectbox("Sort by", options=sort_opts, index=0, key="grand.sort_by")
    ascending = st.checkbox("Ascending", value=False, key="grand.asc")
    if sort_by in df2.columns:
        df2[sort_by] = pd.to_numeric(df2[sort_by], errors="coerce")
        df2 = df2.sort_values(sort_by, ascending=bool(ascending))

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
    show_cols = [c for c in show_cols if c in df2.columns]

    # Verdict visibility toggles
    vc1, vc2, vc3 = st.columns(3)
    with vc1:
        show_pass = st.checkbox("Show PASS", value=True, key="grand.show_pass")
    with vc2:
        show_warn = st.checkbox("Show WARN", value=True, key="grand.show_warn")
    with vc3:
        show_fail = st.checkbox("Show FAIL/UNMEASURED", value=False, key="grand.show_fail")

    mask_v = []
    for v in df2.get("grand.verdict", pd.Series([], dtype=str)).astype(str):
        if v == "PASS" and show_pass:
            mask_v.append(True)
        elif v == "WARN" and show_warn:
            mask_v.append(True)
        elif v in {"FAIL", "UNMEASURED"} and show_fail:
            mask_v.append(True)
        else:
            mask_v.append(False)

    df_show = df2[pd.Series(mask_v, index=df2.index)] if len(mask_v) == len(df2) else df2

    st.dataframe(df_show[show_cols], use_container_width=True, height=520)

    st.download_button(
        "Download grand survivors (CSV)",
        data=df2.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_run_name}_grand_survivors.csv",
    )

    # =============================================================================
    # Deep dive (grand)
    # =============================================================================
    st.divider()
    st.subheader("Deep dive")

    if df2.empty:
        st.info("No grand survivors under the current rules. Relax constraints or run RS/WF.")
        st.stop()

    pick = st.selectbox(
        "Pick a strategy",
        options=df2["config_id"].astype(str).tolist()[:5000],
        index=0,
        key="deep.pick",
    )
    if not pick:
        st.stop()

    # Row for this strategy (for stage receipts)
    row = df2[df2["config_id"].astype(str) == str(pick)].iloc[0].to_dict()

    # Stage receipts: show verdict + reasons
    st.write("#### Receipts (why this passed/warned/failed)")

    def _stage_receipt(title: str, verdict_key: str, q_fn, ans: Dict[str, int]) -> None:
        out = evaluate_row_with_questions(row, q_fn(), ans)
        badge = out.verdict
        st.markdown(f"**{title}: `{badge}`**  —  {out.crits} crit, {out.warns} warn, {out.missing} missing")
        if out.violations:
            # Show a compact table
            vdf = pd.DataFrame(out.violations)
            keep = [c for c in ["severity", "metric", "value", "op", "threshold", "message"] if c in vdf.columns]
            st.dataframe(vdf[keep], use_container_width=True, height=220)
        elif out.missing_metrics:
            st.caption("No violations, but some metrics were missing for this stage.")
            st.code(", ".join(out.missing_metrics))
        else:
            st.caption("No violations.")

    with st.expander("Batch receipts", expanded=False):
        _stage_receipt("Batch", "batch.verdict", batch_questions, batch_ans)
    if rs_sum is not None and not rs_sum.empty:
        with st.expander("Rolling Starts receipts", expanded=False):
            _stage_receipt("Rolling Starts", "rsq.verdict", rolling_questions, rs_ans)
    if wf_sum is not None and not wf_sum.empty:
        with st.expander("Walkforward receipts", expanded=False):
            _stage_receipt("Walkforward", "wfq.verdict", walkforward_questions, wf_ans)

    # Load config details
    cfg_map = {r.get("config_id"): r.get("normalized") for r in _load_jsonl(run_dir / "configs_resolved.jsonl")}
    cfg_norm = cfg_map.get(str(pick), {})
    if cfg_norm:
        with st.expander("Config (normalized)", expanded=False):
            st.json(cfg_norm)

    # --- Replay artifacts (top-k OR on-demand cache) ---
    art_dir = top_map.get(str(pick))
    if not (art_dir and art_dir.exists()):
        cache_dir = run_dir / "replay_cache" / str(pick)
        if cache_dir.exists():
            art_dir = cache_dir

    if art_dir and art_dir.exists():
        eq_path = art_dir / "equity_curve.csv"
        cfg_path = art_dir / "config.json"
        met_path = art_dir / "metrics.json"
        tr_path = art_dir / "trades.csv"
        fi_path = art_dir / "fills.csv"
        if cfg_path.exists():
            with st.expander("Config (artifact config.json)", expanded=False):
                st.json(_read_json(cfg_path))
        if eq_path.exists():
            eq = _load_csv(eq_path)
            if eq is not None and not eq.empty:
                if "dt" in eq.columns:
                    eq["dt"] = pd.to_datetime(eq["dt"], errors="coerce", utc=True)
                if px is not None and "equity" in eq.columns:
                    fig = px.line(eq, x="dt" if "dt" in eq.columns else None, y="equity", title="Equity curve (Batch replay)")
                    st.plotly_chart(fig, use_container_width=True)
                st.download_button("Download equity_curve.csv", data=eq_path.read_bytes(), file_name=f"{pick}_equity_curve.csv")

        cdl1, cdl2, cdl3 = st.columns(3)
        with cdl1:
            if met_path.exists():
                st.download_button("Download metrics.json", data=met_path.read_bytes(), file_name=f"{pick}_metrics.json")
        with cdl2:
            if tr_path.exists():
                st.download_button("Download trades.csv", data=tr_path.read_bytes(), file_name=f"{pick}_trades.csv")
        with cdl3:
            if fi_path.exists():
                st.download_button("Download fills.csv", data=fi_path.read_bytes(), file_name=f"{pick}_fills.csv")
    else:
        st.info("No saved artifacts for this config yet. Generate a replay to inspect it.")
        replay_script = REPO_ROOT / "tools" / "generate_replay_artifacts.py"
        if replay_script.exists():
            if st.button("Generate replay artifacts", key="replay.gen"):
                cmd = [PY, str(replay_script), "--from-run", str(run_dir), "--config-id", str(pick)]
                _run_cmd(cmd, cwd=REPO_ROOT, label="Replay: generate artifacts")
                st.rerun()
        else:
            st.warning("Missing tools/generate_replay_artifacts.py (add it to enable replay generation).")

    # Rolling detail plot (lightweight)
    if rs_dir_effective and (rs_dir_effective / "rolling_starts_detail.csv").exists():
        rs_det = load_rs_detail(run_dir, rs_dir_effective)
        if rs_det is not None and not rs_det.empty and "config_id" in rs_det.columns:
            d = rs_det[rs_det["config_id"].astype(str) == str(pick)].copy()
            if not d.empty and px is not None and "performance.twr_total_return" in d.columns:
                d["start_dt"] = pd.to_datetime(d.get("start_dt"), errors="coerce", utc=True)
                fig = px.histogram(d, x="performance.twr_total_return", nbins=40, title="Rolling-start TWR distribution (detail)")
                st.plotly_chart(fig, use_container_width=True)

    # Walkforward plot (lightweight)
    if wf_dir_effective and (wf_dir_effective / "wf_results.csv").exists():
        wf_rows = load_wf_results(wf_dir_effective)
        if wf_rows is not None and not wf_rows.empty and "config_id" in wf_rows.columns:
            d = wf_rows[wf_rows["config_id"].astype(str) == str(pick)].copy()
            if not d.empty and px is not None:
                if "window_return" in d.columns:
                    fig = px.line(d, x="window_idx", y="window_return", title="Walkforward window returns (detail)")
                    st.plotly_chart(fig, use_container_width=True)

