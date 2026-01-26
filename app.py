from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from lab.eval import evaluate_profile
from lab.metrics import METRICS
from lab.profile_default import default_questions

REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "runs"
DATA_DIR = REPO_ROOT / "data"


def _latest_batch_run(runs_dir: Path) -> Optional[Path]:
    xs = [p for p in runs_dir.glob("batch_*") if p.is_dir()]
    if not xs:
        return None
    return sorted(xs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _run(cmd: list[str], cwd: Path) -> None:
    st.code(" ".join(cmd))
    res = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        st.error(res.stdout)
        st.error(res.stderr)
        raise RuntimeError("Command failed")
    if res.stdout:
        st.text(res.stdout.strip())
    if res.stderr:
        # tqdm prints to stderr sometimes; show it but don't treat as fatal
        st.text(res.stderr.strip())

@st.cache_data(show_spinner=False)

def _read_csv_cached(path: str, mtime: float) -> pd.DataFrame:
    # mtime is a cache buster
    _ = mtime
    return pd.read_csv(path)

def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return _read_csv_cached(str(path), path.stat().st_mtime)

def _severity_rank(s: str) -> int:
    s = str(s or "").strip().lower()
    if s == "critical" or s == "crit":
        return 2
    if s == "warn" or s == "warning":
        return 1
    return 0

st.set_page_config(page_title="DCA/Swing Lab", layout="wide")

st.title("DCA/Swing Stress Test Lab (Daily, Spot, Long-Only)")

with st.sidebar:
    st.header("1) Data")
    uploaded = st.file_uploader("Upload daily OHLCV CSV", type=["csv"])

    sample_path = DATA_DIR / "eth_daily_2023_to_now.csv"
    use_sample = st.checkbox(
        f"Use sample dataset ({sample_path.name})", value=(uploaded is None)
    )

    st.header("2) Grid")
    n = st.selectbox("Variants", options=[250, 500, 1000], index=2)
    seed = st.number_input("Seed", min_value=1, max_value=10_000_000, value=1, step=1)

    st.header("3) Batch settings")
    jobs = st.selectbox("Jobs (Windows: start with 1)", options=[1, 2, 4], index=0)
    rerun_n = st.selectbox("Rerun N", options=[50, 100, 200], index=2)
    top_k = st.selectbox("Save top K artifacts", options=[0, 10, 20], index=2)

    st.header("4) Ranking / Filters")
    max_dd = st.slider("Max equity drawdown", 0.0, 0.9, 0.35, 0.01)
    score = st.selectbox(
        "Score",
        options=["calmar_equity", "profit_dd", "twr_dd", "profit"],
        index=0,
    )

    st.header("6) Your comfort settings")
    st.caption(
        "Answer in plain English. We will flag any strategies that violate the limits "
        "you choose. No hype, just evidence."
    )

    questions = default_questions()
    answers: dict[str, int] = {}
    for q in questions:
        labels = [c.label for c in q.choices]
        key = f"pref.{q.id}"
        if key not in st.session_state:
            st.session_state[key] = int(q.default_index)

        idx = st.selectbox(
            q.title,
            options=list(range(len(labels))),
            format_func=lambda i: labels[int(i)],
            index=int(q.default_index),
            key=key,
            help=q.explanation,
        )
        answers[q.id] = int(idx)

    st.header("5) Rolling starts")
    wf_top_n = st.selectbox("Rolling-start top N", options=[10, 20, 50], index=1)
    start_step = st.selectbox("Start step (bars)", options=[30, 60, 90], index=0)
    min_bars = st.selectbox("Min bars per run", options=[365, 540, 720], index=0)

# Resolve dataset path
tmp_data_dir = RUNS_DIR / "_ui_data"
tmp_data_dir.mkdir(parents=True, exist_ok=True)

data_path: Optional[Path] = None
if use_sample:
    if not sample_path.exists():
        st.error(f"Sample dataset missing: {sample_path}")
        st.stop()
    data_path = sample_path
else:
    if uploaded is None:
        st.warning("Upload a CSV or select the sample dataset.")
        st.stop()
    out = tmp_data_dir / f"upload_{int(time.time())}.csv"
    out.write_bytes(uploaded.getvalue())
    data_path = out

assert data_path is not None

st.caption(f"Using data: {data_path}")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    run_batch = st.button("Run sweep + rerun", type="primary")
with col2:
    run_rs = st.button("Run rolling-starts (top N)")
with col3:
    st.write("")

if run_batch:
    t0 = time.time()

    # 1) generate grid
    grid_path = tmp_data_dir / f"dca_grid_{n}_seed{seed}.jsonl"
    _run(
        [
            "python",
            str(REPO_ROOT / "tools" / "make_dca_grid.py"),
            "--out",
            str(grid_path),
            "--n",
            str(n),
            "--seed",
            str(seed),
        ],
        cwd=REPO_ROOT,
    )

    # 2) run batch
    _run(
        [
            "python",
            "-m",
            "engine.batch",
            "--data",
            str(data_path),
            "--grid",
            str(grid_path),
            "--template",
            "strategies.dca_swing:Strategy",
            "--market-mode",
            "spot",
            "--jobs",
            str(jobs),
            "--fast-sweep",
            "--min-trades",
            "0",
            "--max-best-over-wins",
            "999",
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
        ],
        cwd=REPO_ROOT,
    )

    # 3) postprocess latest batch run
    latest = _latest_batch_run(RUNS_DIR)
    if latest is None:
        st.error("No batch_* run folder found.")
        st.stop()

    _run(
        [
            "python",
            str(REPO_ROOT / "tools" / "postprocess_batch_results.py"),
            "--from-run",
            str(latest),
            "--score",
            str(score),
            "--max-dd",
            str(max_dd),
            "--top-n",
            str(wf_top_n),
        ],
        cwd=REPO_ROOT,
    )

    st.success(f"Done in {time.time() - t0:.1f}s")

# Load latest run outputs (if any)
latest = _latest_batch_run(RUNS_DIR)
if latest is None:
    st.info("Run a sweep to generate results.")
    st.stop()

post_dir = latest / "post"
ranked_path = post_dir / "ranked.csv"
full_path = latest / "results_full.csv"
top_path = post_dir / "top.csv"
ids_path = post_dir / "top_ids.txt"

st.subheader(f"Latest batch run: {latest.name}")
st.write(str(latest))

ranked = _load_csv(ranked_path)
full = _load_csv(full_path)

rs_dir = latest / "rolling_starts"
rs_sum = rs_dir / "rolling_starts_summary.csv"
rs_det = rs_dir / "rolling_starts_detail.csv"
roll_sum = _load_csv(rs_sum)

if ranked is None:
    st.warning("No ranked.csv found yet. Run sweep + rerun.")
    st.stop()

if full is None:
    st.warning("No results_full.csv found yet. Run sweep + rerun.")
    st.stop()

# Merge ranked + full (ranked already includes many full cols, but keep full as source of truth)
df = full.copy()

# Attach score columns from ranked (score.* and pareto.*)
for c in ranked.columns:
    if c.startswith("score.") or c.startswith("pareto."):
        if c in df.columns:
            continue
        # join by config.id
        tmp = ranked[["config.id", c]].copy() if "config.id" in ranked.columns else None
        if tmp is not None:
            df = df.merge(tmp, on="config.id", how="left")

# Attach rolling-start summary if present
if roll_sum is not None and "config_id" in roll_sum.columns:
    df = df.merge(roll_sum, left_on="config.id", right_on="config_id", how="left")

# Evaluate preference alerts row-by-row
rows = df.to_dict(orient="records")
alerts_reason: list[str] = []
viol_count: list[int] = []
viol_max: list[str] = []
viol_blob: list[object] = []
evid_count: list[int] = []
evid_blob: list[object] = []

for r in rows:
    res = evaluate_profile(r, questions, answers)
    alerts_reason.append(res["alerts.top_reason"] or "")
    viol_count.append(int(res["violations.count"] or 0))
    viol_max.append(res["violations.max_severity"] or "")
    viol_blob.append(res["violations.list"])
    evid_count.append(int(res["evidence.count"] or 0))
    evid_blob.append(res["evidence.list"])

df["violations.count"] = viol_count
df["violations.max_severity"] = viol_max
df["alerts.top_reason"] = alerts_reason    
df["violations.list"] = viol_blob
df["evidence.count"] = evid_count
df["evidence.list"] = evid_blob

def _status(vmax: str, vcount: int, ecount: int) -> str:
    if int(vcount) > 0:
        if str(vmax).lower() == "critical":
            return "CRIT"
        return "WARN"
    if int(ecount) > 0:
        return "DATA"
    return "OK"

df["alerts.status"] = [
    _status(vmax, vc, ec)
    for vmax, vc, ec in zip(
        df["violations.max_severity"], df["violations.count"], df["evidence.count"]
    )
]

df["alerts.rank"] = (
    df["alerts.status"]
    .map({"OK": 0, "DATA": 1, "WARN": 2, "CRIT": 3})
    .fillna(9)
)

st.subheader("Scoreboard (filtered by your comfort settings)")
st.caption(
    "Compact by design. You can reveal all columns in power-user mode. "
    "Strategies marked DATA need rolling-start evidence for some preferences."
)

# Rolling-start coverage banner
total_cfgs = int(len(df))
has_roll = ("windows" in df.columns) and df["windows"].notna().any()
if has_roll:
    cov = int(df["windows"].notna().sum())
    st.info(
        f"Rolling-start coverage: {cov}/{total_cfgs} configs have rolling-start stats."
    )
else:
    st.warning(
        "Rolling-start stats not found for this run. Run rolling-starts to measure "
        "worst-case behavior."
    )


def _status(vmax: str, vcount: int, ecount: int) -> str:
    if int(vcount) > 0:
        if str(vmax).lower() == "critical":
            return "CRIT"
        return "WARN"
    if int(ecount) > 0:
        return "DATA"
    return "OK"


df["alerts.status"] = [
    _status(vmax, vc, ec)
    for vmax, vc, ec in zip(
        df["violations.max_severity"], df["violations.count"], df["evidence.count"]
    )
]

df["alerts.rank"] = (
    df["alerts.status"]
    .map({"OK": 0, "DATA": 1, "WARN": 2, "CRIT": 3})
    .fillna(9)
)

# Create display columns for percent metrics (Streamlit formatting doesn't multiply)
def _pct_col(src: str, dst: str) -> None:
    if src in df.columns:
        x = pd.to_numeric(df[src], errors="coerce")
        df[dst] = x * 100.0

_pct_col("twr_p10", "Worst 10% return (%)")
_pct_col("twr_p50", "Median return (%)")
_pct_col("dd_p90", "High drawdown (p90, %)")
_pct_col("performance.max_drawdown_equity", "Max DD single-run (%)")
_pct_col("performance.twr_total_return", "Single-run TWR (%)")

# --- Underwater + Utilization display columns (if present) ---
if "uw_p90_days" in df.columns:
    df["Underwater (p90 days)"] = pd.to_numeric(
        df["uw_p90_days"], errors="coerce"
    )

if "util_p50" in df.columns:
    df["Utilization (p50, %)"] = (
        pd.to_numeric(df["util_p50"], errors="coerce") * 100.0
    )

# Money display columns
if "equity.net_profit_ex_cashflows" in df.columns:
    df["Net profit (ex deposits)"] = pd.to_numeric(
        df["equity.net_profit_ex_cashflows"], errors="coerce"
    )

# Optional: show turnover only if fee preference question exists (it does)
if "exposure.turnover_notional_over_avg_equity" in df.columns:
    df["Turnover (x over test)"] = pd.to_numeric(
        df["exposure.turnover_notional_over_avg_equity"], errors="coerce"
    )

score_col = f"score.{score}"
if score_col in df.columns:
    df["Score"] = pd.to_numeric(df[score_col], errors="coerce")

show_only_ok = st.checkbox(
    "Show only strategies that fit my comfort settings (OK only)",
    value=False,
)
show_power = st.checkbox("Show all columns (power user)", value=False)

view = df.copy()
if show_only_ok:
    view = view[view["alerts.status"] == "OK"].copy()

# Sort: best evidence/lowest severity first, then score desc
sort_cols = ["alerts.rank", "violations.count", "evidence.count"]
ascending = [True, True, True]

if "Score" in view.columns:
    sort_cols.append("Score")
    ascending.append(False)

view = view.sort_values(sort_cols, ascending=ascending)

# Compact scoreboard columns
scoreboard_cols = [
    "alerts.status",
    "violations.count",
    "evidence.count",
    "alerts.top_reason",
    "config.id",
]
if "Score" in view.columns:
    scoreboard_cols.append("Score")

# Prefer rolling-start metrics if present; otherwise single-run fallbacks still show
for c in ["Worst 10% return (%)", "Median return (%)", "High drawdown (p90, %)"]:
    if c in view.columns:
        scoreboard_cols.append(c)

if "Underwater (p90 days)" in view.columns:
    scoreboard_cols.append("Underwater (p90 days)")

if "Utilization (p50, %)" in view.columns:
    scoreboard_cols.append("Utilization (p50, %)")

scoreboard_cols += [
    "Net profit (ex deposits)",
    "Max DD single-run (%)",
    "Turnover (x over test)",
]
scoreboard_cols = [c for c in scoreboard_cols if c in view.columns]

# Column formatting
colcfg = {
    "alerts.status": st.column_config.TextColumn("Status", width="small"),
    "violations.count": st.column_config.NumberColumn("Violations", width="small"),
    "evidence.count": st.column_config.NumberColumn("Evidence", width="small"),
    "alerts.top_reason": st.column_config.TextColumn("Why flagged", width="large"),
    "config.id": st.column_config.TextColumn("Config ID", width="medium"),
}
if "Score" in view.columns:
    colcfg["Score"] = st.column_config.NumberColumn("Score", format="%.4f")

for c in ["Worst 10% return (%)", "Median return (%)", "High drawdown (p90, %)", "Max DD single-run (%)", "Single-run TWR (%)"]:
    if c in view.columns:
        colcfg[c] = st.column_config.NumberColumn(c, format="%.2f%%")

if "Net profit (ex deposits)" in view.columns:
    colcfg["Net profit (ex deposits)"] = st.column_config.NumberColumn(
        "Net profit (ex deposits)", format="$%.2f"
    )

if "Turnover (x over test)" in view.columns:
    colcfg["Turnover (x over test)"] = st.column_config.NumberColumn(
        "Turnover (x over test)", format="%.2f"
    )

if "Underwater (p90 days)" in view.columns:
    colcfg["Underwater (p90 days)"] = st.column_config.NumberColumn(
        "Underwater (p90 days)",
        format="%.0f",
    )

if "Utilization (p50, %)" in view.columns:
    colcfg["Utilization (p50, %)"] = st.column_config.NumberColumn(
        "Utilization (p50, %)",
        format="%.2f%%",
    )

if show_power:
    st.dataframe(view, use_container_width=True, height=420)
else:
    st.dataframe(
        view[scoreboard_cols],
        use_container_width=True,
        height=420,
        column_config=colcfg,
    )

# Deep dive: pick a config to explain alerts
st.subheader("Deep dive")
cfg_ids = view["config.id"].astype(str).tolist() if "config.id" in view.columns else []
if cfg_ids:
    pick = st.selectbox("Select a config.id", options=cfg_ids[:500])
    row = view[view["config.id"].astype(str) == str(pick)].iloc[0].to_dict()

    st.write(
        f"Status: **{row.get('alerts.status','OK')}**  "
        f"(violations={int(row.get('violations.count',0))}, evidence={int(row.get('evidence.count',0))})"
    )

    viols = row.get("violations.list", []) or []
    evid = row.get("evidence.list", []) or []

    if len(viols) == 0:
        st.success("No violations detected for your selected comfort settings.")
    else:
        st.write("#### Violations")
        for a in viols:
            spec = METRICS.get(a.metric_id)
            label = spec.label if spec else a.metric_id
            st.error(a.message) if a.severity == "critical" else st.warning(a.message)

    if len(evid) > 0:
        st.write("#### Evidence notes (not violations)")
        for a in evid:
            st.info(a.message)

    # Rolling-start metrics for this config (if available in merged df)
    st.write("#### Rolling-start snapshot (this config)")
    snap_cols = [
        "twr_p10",
        "twr_p50",
        "twr_p90",
        "dd_p90",
        "profit_p10",
        "profit_p50",
        "profit_p90",
        "robustness_score",
        "windows",
        "uw_p90_days",
        "util_p50",
    ]
    snap = {k: row.get(k, None) for k in snap_cols if k in row}
    if snap:
        st.write(pd.DataFrame([snap]))
    else:
        st.caption("No rolling-start summary found for this config yet.")

    # Optional: show saved artifact path if it exists in top/
    top_dir = latest / "top"
    if top_dir.exists():
        matches = list(top_dir.glob(f"*_{pick}"))
        if matches:
            st.info(f"Artifact folder: {matches[0]}")

else:
    st.info("No configs available in current view.")

if run_rs:
    if not ids_path.exists():
        st.error("Missing top_ids.txt; run postprocess first.")
        st.stop()

    _run(
        [
            "python",
            "-m",
            "research.rolling_starts",
            "--from-run",
            str(latest),
            "--top-n",
            str(wf_top_n),
            "--start-step",
            str(start_step),
            "--min-bars",
            str(min_bars),
            "--seed",
            "1",
            "--starting-equity",
            "1000",
        ],
        cwd=REPO_ROOT,
    )

if rs_sum.exists():
    st.subheader("Rolling-start sensitivity (summary)")
    df_rs = _load_csv(rs_sum)
    if df_rs is None:
        st.stop()
    df_rs = df_rs.sort_values("robustness_score", ascending=False)
    st.dataframe(df_rs, use_container_width=True, height=420)

    st.download_button(
        "Download rolling_starts_summary.csv",
        data=rs_sum.read_bytes(),
        file_name="rolling_starts_summary.csv",
    )
if rs_det.exists():
    st.download_button(
        "Download rolling_starts_detail.csv",
        data=rs_det.read_bytes(),
        file_name="rolling_starts_detail.csv",
    )