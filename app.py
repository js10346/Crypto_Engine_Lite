
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st

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


def _run_cmd(cmd: List[str], *, cwd: Path, label: str) -> None:
    """
    Run a command and stream stdout/stderr into the UI.
    """
    if not cmd:
        raise ValueError("Empty command")

    # Make "python" consistent across platforms
    if str(cmd[0]).lower() in {"python", "py", "py.exe", "python3"}:
        cmd = [PY, *cmd[1:]]

    with st.expander(label, expanded=True):
        st.code(" ".join(cmd), language="bash")
        t0 = time.time()
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
        dt = time.time() - t0

        if p.stdout:
            st.code(p.stdout, language="text")
        if p.stderr:
            st.code(p.stderr, language="text")

        if p.returncode != 0:
            raise RuntimeError(f"Command failed (code={p.returncode}) after {dt:.1f}s")


def _list_runs() -> List[Path]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    xs = [p for p in RUNS_DIR.glob("batch_*") if p.is_dir()]
    return sorted(xs, key=lambda p: p.stat().st_mtime, reverse=True)


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
    # and are not currently in lab.metrics, so we keep labels simple here.
    return [
        QuestionSpec(
            id="wf_median",
            title="Do you require the typical walk-forward window to be positive?",
            explanation="Median return across walk-forward windows.",
            choices=[
                ChoiceSpec("Yes (median window return ≥ 0)", [ConstraintSpec("median_window_return", ">=", 0.0, "warn")]),
                ChoiceSpec("No", []),
            ],
            default_index=0,
        ),
        QuestionSpec(
            id="wf_min",
            title="How bad can the worst walk-forward window be?",
            explanation="Minimum window return across walk-forward windows.",
            choices=[
                ChoiceSpec("Worst window must be ≥ -10%", [ConstraintSpec("min_window_return", ">=", -0.10, "warn")]),
                ChoiceSpec("Worst window must be ≥ -25%", [ConstraintSpec("min_window_return", ">=", -0.25, "info")]),
                ChoiceSpec("Don't filter on this", []),
            ],
            default_index=1,
        ),
        QuestionSpec(
            id="wf_consistency",
            title="How consistent should it be across windows?",
            explanation="Percent of windows with positive return.",
            choices=[
                ChoiceSpec("At least 60% of windows profitable", [ConstraintSpec("pct_profitable_windows", ">=", 0.60, "warn")]),
                ChoiceSpec("At least 50% of windows profitable", [ConstraintSpec("pct_profitable_windows", ">=", 0.50, "info")]),
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


# =============================================================================
# UI: left rail (runs + mode)
# =============================================================================

with st.sidebar:
    st.header("Runs")
    runs = _list_runs()
    run_names = [p.name for p in runs]
    default_idx = 0

    # Persist selection across reruns
    if "selected_run" not in st.session_state:
        st.session_state["selected_run"] = run_names[0] if run_names else ""

    open_existing = st.selectbox(
        "Open existing run",
        options=["(new run)"] + run_names,
        index=(1 + run_names.index(st.session_state["selected_run"]) if st.session_state["selected_run"] in run_names else 0),
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

    if "ui.stage" not in st.session_state:
        st.session_state["ui.stage"] = "batch"

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
                    str(max(25, min(200, n))),
                    "--seed",
                    str(seed),
                ]
                if mode.startswith("Stress test"):
                    grid_cmd += [
                        "--mode",
                        "neighborhood",
                        "--base",
                        str(base_path),
                        "--include-base",
                        "true",
                        "--width",
                        str(width),
                        "--vary",
                        ",".join(vary),
                    ]
                else:
                    grid_cmd += ["--mode", "random", "--include-base", "true"]

                _run_cmd(grid_cmd, cwd=REPO_ROOT, label="Generate preview grid")
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
                    str(int(st.session_state["new.grid_n"])),
                    "--seed",
                    str(int(st.session_state["new.grid_seed"])),
                ]
                if str(st.session_state.get("new.grid_mode2", "")).startswith("Stress test"):
                    grid_cmd += [
                        "--mode",
                        "neighborhood",
                        "--base",
                        str(base_path),
                        "--include-base",
                        "true",
                        "--width",
                        str(st.session_state.get("new.grid_width", "medium")),
                        "--vary",
                        ",".join(st.session_state.get("new.grid_vary", ["deposits","buys"])),
                    ]
                else:
                    grid_cmd += ["--mode", "random", "--include-base", "true"]

                _run_cmd(grid_cmd, cwd=REPO_ROOT, label="1) Generate variants grid")

                # 3) Batch
                batch_cmd: List[str] = [
                    PY,
                    "-m",
                    "engine.batch",
                    "--data",
                    str(data_path),
                    "--grid",
                    str(grid_path),
                    "--template",
                    str(st.session_state.get("new.template_path", "strategies.dca_swing:Strategy")),
                    "--market-mode",
                    "spot",
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
                _run_cmd(batch_cmd, cwd=REPO_ROOT, label="2) Batch sweep + rerun")

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
                st.session_state["ui.open_run"] = run_dir.name  # best effort

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
meta = _read_json(run_dir / "batch_meta.json")
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

    # Table
    cols = [
        "config_id",
        "config.label",
        "batch.verdict",
        "equity.net_profit_ex_cashflows",
        "performance.twr_total_return",
        "performance.max_drawdown_equity",
        "trades_summary.trades_closed",
    ]
    # Optional score columns
    for c in ["score.calmar_equity", "score.profit_dd", "score.twr_dd", "score.profit"]:
        if c in df_show.columns and c not in cols:
            cols.append(c)

    cols = [c for c in cols if c in df_show.columns]
    st.dataframe(df_show[cols], use_container_width=True, height=520)

    st.download_button(
        "Download batch survivors (CSV)",
        data=df_show.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_run_name}_batch_view.csv",
    )

    st.divider()
    st.write("Next: run Rolling Starts to measure start-date fragility.")

# =============================================================================
# Stage B: Rolling Starts
# =============================================================================

elif stage_pick == "rs":
    st.write("### B) Rolling Starts (start-date sensitivity)")
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
        preset = st.selectbox("Preset", options=["Quick", "Standard", "Thorough"], index=0, key="rs.preset")

        if preset == "Quick":
            start_step = 90
            min_bars = 365
        elif preset == "Standard":
            start_step = 30
            min_bars = 365
        else:
            start_step = 15
            min_bars = 365

        start_step = int(st.number_input("Start step (bars)", 1, 365, int(start_step), 5, key="rs.start_step"))
        min_bars = int(st.number_input("Min bars per start", 30, 5000, int(min_bars), 30, key="rs.min_bars"))

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
                str(min_bars),
                "--seed",
                "1",
                "--starting-equity",
                str(float(meta.get("starting_equity", 1000.0) or 1000.0)),
            ]
            _run_cmd(cmd, cwd=REPO_ROOT, label="Rolling Starts")
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

    cols = [
        "config_id",
        "config.label",
        "rsq.verdict",
        "twr_p10",
        "twr_p50",
        "dd_p90",
        "uw_p90_days",
        "util_p50",
        "robustness_score",
        "windows",
    ]
    cols = [c for c in cols if c in df_show.columns]
    if "rs.measured" in df_show.columns:
        cols.insert(2, "rs.measured")
    st.dataframe(df_show[cols], use_container_width=True, height=520)

    st.download_button(
        "Download rolling-start view (CSV)",
        data=df_show.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_run_name}_rolling_view.csv",
    )

    if rs_det is not None and not rs_det.empty:
        st.caption("Detail file detected (rolling_starts_detail.csv). Deep dive will plot per-start distribution.")

# =============================================================================
# Stage C: Walkforward
# =============================================================================

elif stage_pick == "wf":
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

    # Choose WF run dir
    left, right = st.columns([2, 1])

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

    with right:
        st.write("**Quick presets**")
        preset = st.selectbox("Preset", options=["Quick", "Standard", "Thorough"], index=0, key="wf.preset")
        if preset == "Quick":
            window_days = 30
            step_days = 30
            min_bars = 365
        elif preset == "Standard":
            window_days = 60
            step_days = 30
            min_bars = 365
        else:
            window_days = 90
            step_days = 15
            min_bars = 365

        window_days = int(st.number_input("Window days", 7, 365, int(window_days), 5, key="wf.window_days"))
        step_days = int(st.number_input("Step days", 1, 365, int(step_days), 5, key="wf.step_days"))
        min_bars = int(st.number_input("Min bars", 30, 10_000, int(min_bars), 30, key="wf.min_bars"))
        jobs = int(st.number_input("Jobs", 1, 64, 8, 1, key="wf.jobs"))

    survivors_ids = survivors["config_id"].astype(str).tolist()
    N = len(survivors_ids)

    # WF output dir
    wf_out_dir = wf_root / f"wf_win{window_days}_step{step_days}_min{min_bars}_n{N}"
    st.caption(f"Will run on survivors: {N} configs → output: {wf_out_dir}")

    if st.button("Run Walkforward for all survivors", type="primary", disabled=(N == 0)):
        try:
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
                str(min_bars),
                "--jobs",
                str(jobs),
                "--out",
                str(wf_out_dir),
                "--sort-by",
                "gates.passed",  # stable, non-NaN, includes everyone selected by top-n
                "--sort-desc",
            ]
            _run_cmd(cmd, cwd=REPO_ROOT, label="Walkforward")
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
        "pct_profitable_windows",
        "mean_window_return",
        "median_window_return",
        "min_window_return",
        "max_window_return",
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

# =============================================================================
# Stage D: Grand verdict + deep dive
# =============================================================================

else:
    st.write("### D) Grand verdict (Batch + RS + WF)")

    # Load latest RS/WF if present
    rs_dir_effective = rs_latest
    wf_dir_effective = wf_latest

    rs_sum = load_rs_summary(run_dir, rs_dir_effective) if rs_dir_effective else None
    wf_sum = load_wf_summary(wf_dir_effective) if wf_dir_effective else None

    df = survivors.copy()
    df = _ensure_config_id(df)

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
        # PASS or WARN
        return verdict in {"PASS", "WARN"}

    keep_mask = []
    for _, r in df.iterrows():
        ok = True
        ok = ok and _keep(str(r.get("batch.verdict", "")), req_batch)
        v_rs = str(r.get("rsq.verdict", "UNMEASURED"))
        if v_rs == "UNMEASURED":
            ok = ok and (req_rs == "Ignore")
        else:
            ok = ok and _keep(v_rs, req_rs)
        v_wf = str(r.get("wfq.verdict", "UNMEASURED"))
        if v_wf == "UNMEASURED":
            ok = ok and (req_wf == "Ignore")
        else:
            ok = ok and _keep(v_wf, req_wf)
        keep_mask.append(bool(ok))
    df2 = df[pd.Series(keep_mask, index=df.index)].copy()

    st.success(f"Grand survivors: {len(df2)}/{len(df)}")

    # Ranking
    st.subheader("Ranking")
    sort_opts = []
    if "score.calmar_equity" in df2.columns:
        sort_opts.append("score.calmar_equity")
    if "robustness_score" in df2.columns:
        sort_opts.append("robustness_score")
    if "median_window_return" in df2.columns:
        sort_opts.append("median_window_return")
    sort_opts += ["equity.net_profit_ex_cashflows", "performance.twr_total_return"]

    sort_by = st.selectbox("Sort by", options=sort_opts, index=0, key="grand.sort_by")
    ascending = st.checkbox("Ascending", value=False, key="grand.asc")
    if sort_by in df2.columns:
        df2[sort_by] = pd.to_numeric(df2[sort_by], errors="coerce")
        df2 = df2.sort_values(sort_by, ascending=bool(ascending))

    cols = [
        "config_id",
        "config.label",
        "batch.verdict",
        "rsq.verdict",
        "wfq.verdict",
        "equity.net_profit_ex_cashflows",
        "performance.twr_total_return",
        "performance.max_drawdown_equity",
        "robustness_score",
        "median_window_return",
        "pct_profitable_windows",
    ]
    cols = [c for c in cols if c in df2.columns]
    st.dataframe(df2[cols], use_container_width=True, height=520)

    st.download_button(
        "Download grand survivors (CSV)",
        data=df2.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_run_name}_grand_survivors.csv",
    )

    st.divider()
    st.subheader("Deep dive")

    pick = st.selectbox(
        "Pick a strategy",
        options=df2["config_id"].astype(str).tolist()[:5000],
        index=0 if len(df2) else None,
        key="deep.pick",
    )

    if not pick:
        st.stop()

    # Load config details
    cfg_map = {r.get("config_id"): r.get("normalized") for r in _load_jsonl(run_dir / "configs_resolved.jsonl")}
    cfg_norm = cfg_map.get(str(pick), {})
    if cfg_norm:
        with st.expander("Config (normalized)", expanded=False):
            st.json(cfg_norm)

    # Show artifacts if present
    art_dir = top_map.get(str(pick))
    if art_dir and art_dir.exists():
        eq_path = art_dir / "equity_curve.csv"
        cfg_path = art_dir / "config.json"
        if cfg_path.exists():
            with st.expander("Config (artifact config.json)", expanded=False):
                st.json(_read_json(cfg_path))
        if eq_path.exists():
            eq = _load_csv(eq_path)
            if eq is not None and not eq.empty:
                # dt column expected in artifact
                if "dt" in eq.columns:
                    eq["dt"] = pd.to_datetime(eq["dt"], errors="coerce", utc=True)
                if px is not None and "equity" in eq.columns:
                    fig = px.line(eq, x="dt" if "dt" in eq.columns else None, y="equity", title="Equity curve")
                    st.plotly_chart(fig, use_container_width=True)
                st.download_button(
                    "Download equity_curve.csv",
                    data=eq_path.read_bytes(),
                    file_name=f"{pick}_equity_curve.csv",
                )
    else:
        st.info("No saved artifacts for this config (not in top-k). Increase top_k or add on-demand artifact generation.")

    # Rolling detail plot
    if rs_dir_effective and (rs_dir_effective / "rolling_starts_detail.csv").exists():
        rs_det = load_rs_detail(run_dir, rs_dir_effective)
        if rs_det is not None and not rs_det.empty and "config_id" in rs_det.columns:
            d = rs_det[rs_det["config_id"].astype(str) == str(pick)].copy()
            if not d.empty and px is not None and "performance.twr_total_return" in d.columns:
                d["start_dt"] = pd.to_datetime(d.get("start_dt"), errors="coerce", utc=True)
                fig = px.histogram(d, x="performance.twr_total_return", nbins=40, title="Rolling-start TWR distribution")
                st.plotly_chart(fig, use_container_width=True)

    # Walkforward plot
    if wf_dir_effective and (wf_dir_effective / "wf_results.csv").exists():
        wf_rows = load_wf_results(wf_dir_effective)
        if wf_rows is not None and not wf_rows.empty and "config_id" in wf_rows.columns:
            d = wf_rows[wf_rows["config_id"].astype(str) == str(pick)].copy()
            if not d.empty and px is not None and "equity.total_return" in d.columns:
                fig = px.line(d, x="window_idx", y="equity.total_return", title="Walkforward window returns")
                st.plotly_chart(fig, use_container_width=True)
