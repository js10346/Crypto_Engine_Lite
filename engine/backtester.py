# engine/backtester.py
from __future__ import annotations

import argparse
import importlib
import warnings
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.contracts import (
    AccountState,
    Candle,
    EngineConstraints,
    ExecutionResult,
    OrderType,
    PlanAction,
    PlanUpdate,
    PositionState,
    StrategyContext,
    TakeProfitLevel,
    TradePlan,
)

# --- PATCH: Import Feature Calculator ---
from engine.features import add_features

# ============================================================
# Small math helpers
# ============================================================


def bps_to_frac(bps: float) -> float:
    return bps / 10_000.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sgn(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def round_price_to_tick(price: float, tick: float, side_sign: int) -> Tuple[float, bool]:
    if tick <= 0:
        return price, False
    q = price / tick
    if side_sign > 0:
        out = math.ceil(q) * tick
    elif side_sign < 0:
        out = math.floor(q) * tick
    else:
        out = round(q) * tick
    return out, (abs(out - price) > 1e-12)


def quantize_delta_to_step(delta: float, step: float) -> Tuple[float, bool]:
    if step <= 0:
        return delta, False
    sign = 1.0 if delta > 0 else (-1.0 if delta < 0 else 0.0)
    mag = abs(delta)
    mag_q = math.floor(mag / step) * step
    out = sign * mag_q
    return out, (abs(out - delta) > 1e-12)


def liquidation_price(
    cash: float, avg_entry: float, pos_qty: float, mmr: float
) -> Optional[float]:
    if abs(pos_qty) < 1e-12:
        return None
    denom = mmr * abs(pos_qty) - pos_qty
    if abs(denom) < 1e-12:
        return None
    return (cash - avg_entry * pos_qty) / denom


def long_stop_hit(o: float, l: float, stop_price: float) -> Tuple[bool, float]:
    return (l <= stop_price), min(stop_price, o)


def short_stop_hit(o: float, h: float, stop_price: float) -> Tuple[bool, float]:
    return (h >= stop_price), max(stop_price, o)

def long_stop_hit_mark(
    *,
    mark_o: float,
    mark_l: float,
    stop_price: float,
    last_o: float,
    last_l: float,
) -> Tuple[bool, float]:
    """
    Trigger on mark. Fill reference uses last prices (market fill model adds slip/spread).

    If mark triggers but last never touched the stop level, we conservatively use last_o
    as the market ref (we still stop out because mark hit).
    """
    hit = (mark_o <= stop_price) or (mark_l <= stop_price)
    if not hit:
        return False, last_o
    ref = min(stop_price, last_o) if last_l <= stop_price else last_o
    return True, float(ref)


def short_stop_hit_mark(
    *,
    mark_o: float,
    mark_h: float,
    stop_price: float,
    last_o: float,
    last_h: float,
) -> Tuple[bool, float]:
    hit = (mark_o >= stop_price) or (mark_h >= stop_price)
    if not hit:
        return False, last_o
    ref = max(stop_price, last_o) if last_h >= stop_price else last_o
    return True, float(ref)


# ============================================================
# Feature access helper (Phase 2 performance)
# ============================================================


class FeatureView:
    """
    A lightweight per-bar view over precomputed feature arrays.

    It mimics a dict enough for strategies that do:
      ctx.features.get("rsi_14")

    This avoids allocating a new dict every bar, which matters a lot
    in batch runs.
    """

    __slots__ = ("_arrays", "_i")

    def __init__(self, arrays: Dict[str, np.ndarray]):
        self._arrays = arrays
        self._i = 0

    def set_index(self, i: int) -> None:
        self._i = int(i)

    def get(self, key: str, default: Any = None) -> Any:
        arr = self._arrays.get(key)
        if arr is None:
            return default
        v = float(arr[self._i])
        if not math.isfinite(v):
            return default
        return v


# ============================================================
# Time + reporting helpers
# ============================================================


SECONDS_PER_YEAR = 365.25 * 24 * 3600.0


def _dt64_utc_naive(dt_str: str) -> np.datetime64:
    """
    Avoids: 'no explicit representation of timezones available for np.datetime64'
    Parse as UTC -> drop tz -> convert.
    """
    ts = pd.to_datetime(dt_str, utc=True)
    ts = ts.tz_localize(None)
    return np.datetime64(ts)


def _format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _format_money(x: float) -> str:
    return f"{x:,.2f}"


def _max_drawdown(equity: pd.Series) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    eq = equity.astype(float).to_numpy()
    peak = eq[0]
    mdd = 0.0
    for x in eq:
        if x > peak:
            peak = x
        if peak > 0:
            dd = (peak - x) / peak
            if dd > mdd:
                mdd = dd
    return float(mdd)


def _exit_counts(trades_df: pd.DataFrame) -> Dict[str, int]:
    if trades_df is None or trades_df.empty or "exit_reason" not in trades_df.columns:
        return {}
    td = trades_df.copy()
    td = td[td["exit_dt"].notna()] if "exit_dt" in td.columns else td
    vc = td["exit_reason"].astype(str).value_counts()
    return {str(k): int(v) for k, v in vc.items()}


def _tp_level_counts(trades_df: pd.DataFrame) -> Dict[str, int]:
    if trades_df is None or trades_df.empty or "tps_hit" not in trades_df.columns:
        return {}

    td = trades_df.copy()
    if "exit_dt" in td.columns:
        td = td[td["exit_dt"].notna()]

    if td.empty:
        return {}

    max_levels = 0
    if "tp_levels" in td.columns:
        max_levels = int(
            pd.to_numeric(td["tp_levels"], errors="coerce").fillna(0).max()
        )

    max_hit = int(pd.to_numeric(td["tps_hit"], errors="coerce").fillna(0).max())
    L = max(max_levels, max_hit)

    tps_hit = pd.to_numeric(td["tps_hit"], errors="coerce").fillna(0).astype(int)

    out: Dict[str, int] = {"trades_closed": int(len(td))}
    for k in range(1, L + 1):
        out[f"tp{k}_hit_trades"] = int((tps_hit >= k).sum())
    return out


def _efficiency_stats(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates fee drag and efficiency.
    """
    if trades_df is None or trades_df.empty:
        return {}

    td = (
        trades_df[trades_df["exit_dt"].notna()]
        if "exit_dt" in trades_df.columns
        else trades_df
    )
    if td.empty:
        return {}

    gross = float(pd.to_numeric(td["gross_pnl"], errors="coerce").fillna(0).sum())
    fees = float(pd.to_numeric(td["fees_total"], errors="coerce").fillna(0).sum())
    net = float(pd.to_numeric(td["net_pnl"], errors="coerce").fillna(0).sum())

    # Impact: fees as % of gross profit (or loss).
    # If gross is near zero, this explodes, so handle carefully.
    impact_pct = (fees / abs(gross) * 100.0) if abs(gross) > 1e-9 else 0.0

    return {
        "gross_pnl": gross,
        "total_fees": fees,
        "net_pnl": net,
        "fee_impact_pct": float(impact_pct),
        "avg_trade_fee": fees / len(td) if len(td) > 0 else 0.0,
    }


def _stop_move_stats(trades_df: pd.DataFrame) -> Dict[str, Any]:
    if trades_df is None or trades_df.empty:
        return {}

    td = trades_df.copy()
    if "exit_dt" in td.columns:
        td = td[td["exit_dt"].notna()]

    if td.empty:
        return {}

    def _sum_col(col: str) -> int:
        if col not in td.columns:
            return 0
        return int(pd.to_numeric(td[col], errors="coerce").fillna(0).sum())

    def _count_true(col: str) -> int:
        if col not in td.columns:
            return 0
        return int(
            pd.to_numeric(td[col], errors="coerce").fillna(0).astype(bool).sum()
        )

    moved_trades = _count_true("stop_moved")
    move_events = _sum_col("stop_move_count")
    move_tp_events = _sum_col("stop_move_tp_count")
    move_plan_events = _sum_col("stop_move_plan_count")

    stop_exits = (
        td[td["exit_reason"].astype(str) == "stop"]
        if "exit_reason" in td.columns
        else td.iloc[0:0]
    )
    stop_hits_total = int(len(stop_exits))

    stop_hits_moved = 0
    if len(stop_exits) and "stop_hit_was_moved" in stop_exits.columns:
        stop_hits_moved = int(
            pd.to_numeric(stop_exits["stop_hit_was_moved"], errors="coerce")
            .fillna(0)
            .astype(bool)
            .sum()
        )

    return {
        "trades_closed": int(len(td)),
        "trades_with_any_stop_move": int(moved_trades),
        "stop_move_events": int(move_events),
        "stop_move_tp_events": int(move_tp_events),
        "stop_move_plan_events": int(move_plan_events),
        "stop_hits_total": int(stop_hits_total),
        "stop_hits_after_move": int(stop_hits_moved),
        "stop_hits_original": int(stop_hits_total - stop_hits_moved),
    }


def _trade_stats(trades_df: pd.DataFrame) -> Dict[str, Any]:
    default_stats = {
        "trades_closed": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "profit_factor": 0.0,
        "expectancy": 0.0,
        "sqn": 0.0,
        "avg_r": 0.0,
        "median_pnl": 0.0,
        "avg_pnl": 0.0,
        "best_trade": 0.0,
        "worst_trade": 0.0,
        "avg_duration_min": 0.0,
        "avg_duration_win_min": 0.0,
        "avg_duration_loss_min": 0.0,
        "max_consec_wins": 0,
        "max_consec_losses": 0,
    }

    if trades_df is None or trades_df.empty:
        return default_stats

    td = trades_df.copy()
    if "exit_dt" in td.columns:
        td = td[td["exit_dt"].notna()]

    if td.empty or "net_pnl" not in td.columns:
        return default_stats

    pnl = pd.to_numeric(td["net_pnl"], errors="coerce").fillna(0.0).astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    pf = (
        (wins.sum() / abs(losses.sum()))
        if len(losses)
        else (float("inf") if len(wins) else 0.0)
    )

    # Durations (Win vs Loss split)
    durations = pd.Series([0.0] * len(td), index=td.index)
    if "entry_dt" in td.columns and "exit_dt" in td.columns:
        # Your engine emits dt strings like: "YYYY-MM-DD HH:MM:SS+00:00"
        # Specify format to avoid slow per-element parsing + warnings.
        fmt = "%Y-%m-%d %H:%M:%S%z"
        entry = pd.to_datetime(
            td["entry_dt"], utc=True, errors="coerce", format=fmt, cache=True
        )
        exit_ = pd.to_datetime(
            td["exit_dt"], utc=True, errors="coerce", format=fmt, cache=True
        )
        durations = (exit_ - entry).dt.total_seconds() / 60.0

    df_dur = pd.DataFrame({"pnl": pnl, "dur": durations}).dropna()
    avg_dur_all = float(df_dur["dur"].mean()) if len(df_dur) else 0.0

    wins_dur = df_dur[df_dur["pnl"] > 0]["dur"]
    loss_dur = df_dur[df_dur["pnl"] <= 0]["dur"]

    avg_dur_win = float(wins_dur.mean()) if len(wins_dur) else 0.0
    avg_dur_loss = float(loss_dur.mean()) if len(loss_dur) else 0.0

    # R-Multiples / SQN
    avg_r = 0.0
    sqn = 0.0
    if "net_R" in td.columns:
        rs = pd.to_numeric(td["net_R"], errors="coerce").dropna()
        if len(rs) > 0:
            avg_r = float(rs.mean())
            std_r = float(rs.std(ddof=1)) if len(rs) > 1 else 0.0
            sqn = (
                (math.sqrt(len(rs)) * avg_r / std_r) if std_r > 1e-9 else 0.0
            )

    # Streaks
    signs = pnl.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).to_numpy()
    max_w = 0
    max_l = 0
    cur_w = 0
    cur_l = 0
    for s in signs:
        if s > 0:
            cur_w += 1
            cur_l = 0
        elif s < 0:
            cur_l += 1
            cur_w = 0
        else:
            cur_w = 0
            cur_l = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)

    return {
        "trades_closed": int(len(pnl)),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(pf) if math.isfinite(pf) else float("inf"),
        "expectancy": float(pnl.mean()) if len(pnl) else 0.0,
        "sqn": float(sqn),
        "avg_r": float(avg_r),
        "median_pnl": float(pnl.median()) if len(pnl) else 0.0,
        "avg_pnl": float(pnl.mean()) if len(pnl) else 0.0,
        "best_trade": float(pnl.max()) if len(pnl) else 0.0,
        "worst_trade": float(pnl.min()) if len(pnl) else 0.0,
        "avg_duration_min": float(avg_dur_all),
        "avg_duration_win_min": float(avg_dur_win),
        "avg_duration_loss_min": float(avg_dur_loss),
        "max_consec_wins": int(max_w),
        "max_consec_losses": int(max_l),
    }


def _slice_pnl_stats(td: pd.DataFrame) -> Dict[str, Any]:
    if td is None or td.empty or "net_pnl" not in td.columns:
        return {"n": 0}

    pnl = pd.to_numeric(td["net_pnl"], errors="coerce").fillna(0.0).astype(float)
    return {
        "n": int(len(pnl)),
        "pct_profitable": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "avg_net": float(pnl.mean()) if len(pnl) else 0.0,
        "median_net": float(pnl.median()) if len(pnl) else 0.0,
        "sum_net": float(pnl.sum()) if len(pnl) else 0.0,
    }


def _tp_sl_profitability_report(trades_df: pd.DataFrame) -> Dict[str, Any]:
    if trades_df is None or trades_df.empty:
        return {}

    td = trades_df.copy()
    if "exit_dt" in td.columns:
        td = td[td["exit_dt"].notna()]

    if td.empty or "net_pnl" not in td.columns:
        return {}

    out: Dict[str, Any] = {}

    if "tps_hit" in td.columns:
        max_tp = int(pd.to_numeric(td["tps_hit"], errors="coerce").fillna(0).max())
        tp_levels: Dict[str, Any] = {}
        for k in range(1, max_tp + 1):
            sub = td[
                pd.to_numeric(td["tps_hit"], errors="coerce").fillna(0).astype(int)
                >= k
            ]
            tp_levels[f"TP{k}"] = _slice_pnl_stats(sub)
        out["tp_levels"] = tp_levels

    if "exit_reason" in td.columns:
        by_reason: Dict[str, Any] = {}
        for reason, sub in td.groupby(td["exit_reason"].astype(str)):
            by_reason[str(reason)] = _slice_pnl_stats(sub)
        out["by_exit_reason"] = by_reason

    if "exit_reason" in td.columns and "stop_hit_was_moved" in td.columns:
        stops = td[td["exit_reason"].astype(str) == "stop"]
        if not stops.empty:
            moved_mask = (
                pd.to_numeric(stops["stop_hit_was_moved"], errors="coerce")
                .fillna(0)
                .astype(bool)
            )
            out["stops"] = {
                "moved": _slice_pnl_stats(stops[moved_mask]),
                "original": _slice_pnl_stats(stops[~moved_mask]),
            }

    return out


def _infer_bar_seconds(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty:
        return None
    if "ts" in df.columns:
        ts = (
            pd.to_numeric(df["ts"], errors="coerce")
            .dropna()
            .astype(np.int64)
            .to_numpy()
        )
        if len(ts) < 3:
            return None
        d = np.diff(ts)
        d = d[d > 0]
        if len(d) == 0:
            return None
        med_ms = float(np.median(d))
        return med_ms / 1000.0
    if "dt" in df.columns:
        dt = pd.to_datetime(df["dt"], utc=True, errors="coerce").dropna()
        if len(dt) < 3:
            return None
        d = dt.diff().dt.total_seconds().dropna()
        d = d[d > 0]
        if len(d) == 0:
            return None
        return float(d.median())
    return None


def _build_performance_stats(df: pd.DataFrame, eq_df: pd.DataFrame) -> Dict[str, Any]:
    if eq_df is None or eq_df.empty or "equity" not in eq_df.columns:
        return {}

    eq = eq_df["equity"].astype(float)
    start = float(eq.iloc[0])
    end = float(eq.iloc[-1])
    mdd = _max_drawdown(eq)

    bar_sec = _infer_bar_seconds(df)
    if bar_sec is None or bar_sec <= 0:
        bar_sec = 60.0

    period_sec = max(1.0, bar_sec * max(1, len(eq) - 1))
    bars_per_year = SECONDS_PER_YEAR / bar_sec

    if start > 0 and end > 0:
        total_log = float(math.log(end / start))
        ann_return = float(math.exp(total_log * (SECONDS_PER_YEAR / period_sec)) - 1.0)
    else:
        total_log = 0.0
        ann_return = 0.0

    eq_pos = eq.replace([np.inf, -np.inf], np.nan).astype(float)
    eq_pos = eq_pos.where(eq_pos > 0).ffill().bfill()
    logret = np.log(eq_pos).diff().fillna(0.0)

    ann_vol = (
        float(logret.std(ddof=0) * math.sqrt(bars_per_year)) if len(logret) else 0.0
    )
    downside = logret.where(logret < 0.0, 0.0)
    downside_vol = (
        float(downside.std(ddof=0) * math.sqrt(bars_per_year))
        if len(downside)
        else 0.0
    )

    sharpe = (ann_return / ann_vol) if ann_vol > 1e-12 else 0.0
    sortino = (ann_return / downside_vol) if downside_vol > 1e-12 else 0.0
    calmar = (
        (ann_return / mdd)
        if mdd > 1e-12
        else (float("inf") if ann_return > 0 else 0.0)
    )

    return {
        "bar_seconds": float(bar_sec),
        "bars_per_year": float(bars_per_year),
        "period_seconds": float(period_sec),
        "max_drawdown": float(mdd),
        "annualized_return": float(ann_return),
        "annualized_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar) if math.isfinite(calmar) else float("inf"),
        "total_log_return": float(total_log),
    }

def _build_cashflow_performance_stats(
    df: pd.DataFrame, eq_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Cashflow-aware performance stats (spot/DCA).

    Assumes cashflows are applied at the *start* of each bar (bar open),
    and equity is recorded at bar close.

    We compute a TWR index:
      ratio_t = equity_t / (equity_{t-1} + cashflow_t)
      idx_t = idx_{t-1} * ratio_t
    """
    if eq_df is None or eq_df.empty or "equity" not in eq_df.columns:
        return {}

    eq = pd.to_numeric(eq_df["equity"], errors="coerce").fillna(0.0).astype(float)
    cf = (
        pd.to_numeric(eq_df.get("cashflow", 0.0), errors="coerce")
        .fillna(0.0)
        .astype(float)
    )

    bar_sec = _infer_bar_seconds(df)
    if bar_sec is None or bar_sec <= 0:
        bar_sec = 86_400.0

    bars_per_year = SECONDS_PER_YEAR / float(bar_sec)
    period_sec = max(1.0, float(bar_sec) * max(1, len(eq) - 1))

    # Build TWR ratios/log returns
    ratios = np.ones(len(eq), dtype=np.float64)
    logrets = np.zeros(len(eq), dtype=np.float64)
    idx = np.ones(len(eq), dtype=np.float64)

    e = eq.to_numpy(dtype=np.float64)
    c = cf.to_numpy(dtype=np.float64)

    for t in range(1, len(e)):
        denom = float(e[t - 1] + c[t])
        if denom <= 1e-12 or (not math.isfinite(denom)):
            ratios[t] = 1.0
            logrets[t] = 0.0
            idx[t] = idx[t - 1]
            continue

        r = float(e[t] / denom)
        if (not math.isfinite(r)) or r <= 0:
            r = 1.0
        ratios[t] = r
        logrets[t] = math.log(r)
        idx[t] = idx[t - 1] * r

    twr_total_log = float(logrets.sum())
    twr_total_return = float(idx[-1] - 1.0) if len(idx) else 0.0
    ann_return = float(math.exp(twr_total_log * (SECONDS_PER_YEAR / period_sec)) - 1.0)

    ann_vol = float(np.std(logrets, ddof=0) * math.sqrt(bars_per_year)) if len(logrets) else 0.0
    downside = np.where(logrets < 0.0, logrets, 0.0)
    downside_vol = float(np.std(downside, ddof=0) * math.sqrt(bars_per_year)) if len(downside) else 0.0

    sharpe = (ann_return / ann_vol) if ann_vol > 1e-12 else 0.0
    sortino = (ann_return / downside_vol) if downside_vol > 1e-12 else 0.0

    # Drawdown on TWR index (deposit-neutral) + equity (for reference)
    mdd_twr = _max_drawdown(pd.Series(idx))
    mdd_eq = _max_drawdown(eq)
    calmar = (
        (ann_return / mdd_twr)
        if mdd_twr > 1e-12
        else (float("inf") if ann_return > 0 else 0.0)
    )

    return {
        "bar_seconds": float(bar_sec),
        "bars_per_year": float(bars_per_year),
        "period_seconds": float(period_sec),
        "cashflow_total": float(c.sum()) if len(c) else 0.0,
        "twr_total_return": float(twr_total_return),
        "max_drawdown": float(mdd_twr),
        "max_drawdown_equity": float(mdd_eq),
        "annualized_return": float(ann_return),
        "annualized_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar) if math.isfinite(calmar) else float("inf"),
        "total_log_return": float(twr_total_log),
    }

def _execution_breakdown(fills_df: pd.DataFrame) -> Dict[str, Any]:
    if fills_df is None or fills_df.empty:
        return {}

    f = fills_df.copy()
    if "order_type" not in f.columns:
        return {}

    f["notional"] = f["qty"].abs().astype(float) * f["price"].astype(float)

    total_notional = float(f["notional"].sum())
    total_fees = float(f["fee"].astype(float).sum()) if "fee" in f.columns else 0.0

    mk = f[f["order_type"] == "limit"]
    tk = f[f["order_type"] == "market"]

    mk_notional = float(mk["notional"].sum()) if len(mk) else 0.0
    tk_notional = float(tk["notional"].sum()) if len(tk) else 0.0

    mk_fees = (
        float(mk["fee"].astype(float).sum()) if (len(mk) and "fee" in mk.columns) else 0.0
    )
    tk_fees = (
        float(tk["fee"].astype(float).sum()) if (len(tk) and "fee" in tk.columns) else 0.0
    )

    def wavg(col: str) -> float:
        if col not in f.columns or total_notional <= 1e-12:
            return 0.0
        return float((f[col].astype(float) * f["notional"]).sum() / total_notional)

    avg_slip = wavg("slip_bps")
    avg_spread = wavg("spread_bps")

    return {
        "fills": int(len(f)),
        "total_notional": float(total_notional),
        "total_fees": float(total_fees),
        "maker": {"fills": int(len(mk)), "notional": float(mk_notional), "fees": float(mk_fees)},
        "taker": {"fills": int(len(tk)), "notional": float(tk_notional), "fees": float(tk_fees)},
        "weighted_avg_slip_bps": float(avg_slip),
        "weighted_avg_spread_bps": float(avg_spread),
    }


def _exposure_stats(eq_df: pd.DataFrame, fills_df: pd.DataFrame) -> Dict[str, Any]:
    if eq_df is None or eq_df.empty:
        return {}
    pos = (
        eq_df["pos_qty"].astype(float)
        if "pos_qty" in eq_df.columns
        else pd.Series([0.0] * len(eq_df))
    )
    in_mkt = float((pos.abs() > 1e-12).mean()) if len(pos) else 0.0

    avg_eq = float(eq_df["equity"].astype(float).mean()) if "equity" in eq_df.columns else 0.0
    turnover = 0.0
    if fills_df is not None and not fills_df.empty and avg_eq > 1e-12:
        notional = (fills_df["qty"].abs().astype(float) * fills_df["price"].astype(float)).sum()
        turnover = float(notional / avg_eq)

    return {
        "time_in_market_frac": float(in_mkt),
        "avg_equity": float(avg_eq),
        "turnover_notional_over_avg_equity": float(turnover),
    }


def _save_daily_monthly_returns(out_dir: Path, eq_df: pd.DataFrame) -> None:
    if eq_df is None or eq_df.empty or "dt" not in eq_df.columns or "equity" not in eq_df.columns:
        return

    t = pd.to_datetime(eq_df["dt"], utc=True, errors="coerce").dt.tz_localize(None)
    eq = eq_df["equity"].astype(float)
    tmp = pd.DataFrame({"dt": t, "equity": eq}).dropna()
    if tmp.empty:
        return

    tmp["date"] = tmp["dt"].dt.date
    daily = tmp.groupby("date", as_index=False).agg(equity_close=("equity", "last"))
    daily["daily_return"] = daily["equity_close"].pct_change().fillna(0.0)
    daily.to_csv(out_dir / "daily_returns.csv", index=False)

    tmp["month"] = tmp["dt"].dt.to_period("M").astype(str)
    monthly = tmp.groupby("month", as_index=False).agg(equity_close=("equity", "last"))
    monthly["monthly_return"] = monthly["equity_close"].pct_change().fillna(0.0)
    monthly.to_csv(out_dir / "monthly_returns.csv", index=False)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _write_text(path: Path, s: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


# ============================================================
# Engine config (ONLY physics knobs)
# ============================================================


@dataclass
class BacktestConfig:
    fee_taker: float = 0.0004
    fee_maker: float = 0.0002

    slippage_taker_bps: float = 1.0
    slippage_maker_bps: float = 0.0

    slippage_taker_vol_mult: float = 0.05
    slippage_maker_vol_mult: float = 0.01
    slippage_taker_bps_max: float = 8.0
    slippage_maker_bps_max: float = 2.0

    spread_bps: float = 2.0
    spread_vol_mult: float = 0.00
    spread_bps_max: float = 10.0

    liq_apply_to_spread: bool = True
    liq_apply_to_slippage: bool = True

    tp_limit_fill_prob_base: float = 0.85
    tp_limit_fill_vol_penalty: float = 0.003
    tp_limit_fill_overshoot_bonus: float = 0.001
    tp_limit_fill_prob_min: float = 0.05
    force_tp_limit_fills: bool = False

    # Liquidation realism
    liq_brutal_buffer_frac: float = 0.10

    # Funding realism
    use_real_funding: bool = True

    funding_rate_8h: float = 0.0001
    funding_interval_hours: int = 8
    funding_jitter_max_frac: float = 0.10

    close_open_at_end: bool = True
    market_mode: str = "spot"  # "spot" (default) or "perps"

class PaperSpotBroker:
    """
    Spot broker (long-only, no leverage).

    - cash is quote currency (USD/USDT)
    - pos_qty is base asset units
    - equity = cash + pos_qty * mark_price
    - buys are clamped to available cash (after fees)
    """

    def __init__(
        self, starting_cash: float, constraints: EngineConstraints, cfg: BacktestConfig
    ):
        self.record_fills = True
        self.record_equity_curve = True
        self.cfg = cfg
        self.constraints = constraints

        self.cash = float(starting_cash)
        self.pos_qty = 0.0
        self.avg_entry = 0.0

        self.realized_pnl = 0.0

        self.fees_paid_total = 0.0
        self.fees_paid_taker = 0.0
        self.fees_paid_maker = 0.0

        # Perps-compat fields (spot always 0.0)
        self.funding_net = 0.0
        self.funding_paid = 0.0
        self.funding_received = 0.0
        self.funding_penalty = 0.0

        self.rejected_min_notional = 0
        self.rejected_qty_step = 0
        self.price_tick_rounds = 0

        self.fills: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []

    def set_recording(self, *, record_fills: bool, record_equity_curve: bool) -> None:
        self.record_fills = bool(record_fills)
        self.record_equity_curve = bool(record_equity_curve)

    def unrealized_pnl(self, mark_price: float) -> float:
        if abs(self.pos_qty) < 1e-12:
            return 0.0
        return (mark_price - self.avg_entry) * self.pos_qty

    def equity(self, mark_price: float) -> float:
        return float(self.cash + self.pos_qty * mark_price)

    def _apply_fill(
        self,
        dt: str,
        delta_qty: float,
        fill_price: float,
        fee_rate: float,
        order_type: str,
        slip_bps: float,
        spread_bps: float,
        liq_mult: float,
    ) -> float:
        if abs(delta_qty) < 1e-12:
            return 0.0

        fee = abs(delta_qty) * fill_price * fee_rate

        old_qty = self.pos_qty
        new_qty = old_qty + delta_qty

        if delta_qty > 0:
            # BUY: pay notional + fee from cash
            notional = delta_qty * fill_price
            self.cash -= (notional + fee)

            # Avg entry update (weighted)
            if abs(old_qty) < 1e-12:
                self.avg_entry = fill_price
            else:
                total = old_qty + delta_qty
                if total > 1e-12:
                    self.avg_entry = (old_qty * self.avg_entry + delta_qty * fill_price) / total
            self.pos_qty = new_qty
        else:
            # SELL: receive notional - fee into cash
            sell_qty = abs(delta_qty)
            proceeds = sell_qty * fill_price
            self.cash += (proceeds - fee)

            # Realized pnl tracking (for reporting)
            realized = (fill_price - self.avg_entry) * sell_qty
            self.realized_pnl += realized

            self.pos_qty = new_qty
            if self.pos_qty <= 1e-12:
                self.pos_qty = 0.0
                self.avg_entry = 0.0

        self.fees_paid_total += fee
        if order_type == "market":
            self.fees_paid_taker += fee
        else:
            self.fees_paid_maker += fee

        side = "buy" if delta_qty > 0 else "sell"
        if self.record_fills:
            self.fills.append(
                {
                    "dt": str(dt),
                    "side": side,
                    "qty": float(delta_qty),
                    "price": float(fill_price),
                    "fee": float(fee),
                    "fee_rate": float(fee_rate),
                    "order_type": str(order_type),
                    "slip_bps": float(slip_bps),
                    "spread_bps": float(spread_bps),
                    "liq_mult": float(liq_mult),
                    "pos_qty": float(self.pos_qty),
                    "avg_entry": float(self.avg_entry),
                    "realized_pnl": float(self.realized_pnl),
                    "equity": float(self.equity(fill_price)),
                    "cash": float(self.cash),
                }
            )

        return float(fee)

    def trade_to_target_qty(
        self,
        dt: str,
        target_qty: float,
        ref_price: float,
        order_type: OrderType,
        slip_bps: float,
        spread_bps: float,
        liq_mult: float,
        force_close: bool = False,
    ) -> ExecutionResult:
        if ref_price <= 0:
            return ExecutionResult(attempted=True, filled=False, reject_reason="bad_price")

        # Long-only spot: clamp target >= 0
        target_qty = max(0.0, float(target_qty))

        delta = (-self.pos_qty) if (force_close and abs(target_qty) < 1e-12) else (
            target_qty - self.pos_qty
        )
        if abs(delta) < 1e-12:
            return ExecutionResult(attempted=True, filled=False, reject_reason="no_change")

        # Never sell more than position (spot)
        if delta < 0:
            delta = max(delta, -self.pos_qty)

        fee_rate = self.cfg.fee_taker if order_type == OrderType.MARKET else self.cfg.fee_maker
        side_sign = 1 if delta > 0 else -1

        applied_spread_bps = 0.0
        applied_slip_bps = 0.0
        price_rounded = False

        if order_type == OrderType.LIMIT:
            pre = ref_price
            ticked, price_rounded = round_price_to_tick(
                pre, self.constraints.price_tick, side_sign
            )
            if price_rounded:
                self.price_tick_rounds += 1
            fill_price = ticked
        else:
            applied_spread_bps = float(spread_bps)
            applied_slip_bps = float(slip_bps)

            half_spread = 0.5 * bps_to_frac(applied_spread_bps)
            spread_price = (
                ref_price * (1.0 + half_spread)
                if side_sign > 0
                else ref_price * (1.0 - half_spread)
            )

            slip = bps_to_frac(applied_slip_bps)
            pre = (
                spread_price * (1.0 + slip)
                if side_sign > 0
                else spread_price * (1.0 - slip)
            )

            ticked, price_rounded = round_price_to_tick(
                pre, self.constraints.price_tick, side_sign
            )
            if price_rounded:
                self.price_tick_rounds += 1
            fill_price = ticked

        # Clamp buys to available cash (after fees)
        buy_clamped = False
        if delta > 0:
            denom = float(fill_price) * (1.0 + float(fee_rate))
            if denom > 0:
                max_buy_qty = float(self.cash) / denom
                if max_buy_qty < delta:
                    delta = max_buy_qty
                    buy_clamped = True

        qty_rounded = False
        if not (force_close and abs(target_qty) < 1e-12):
            delta_q, qty_rounded = quantize_delta_to_step(delta, self.constraints.qty_step)
            if abs(delta_q) < 1e-12:
                self.rejected_qty_step += 1
                return ExecutionResult(
                    attempted=True,
                    filled=False,
                    reject_reason="qty_step",
                    qty_rounded_to_step=qty_rounded,
                    price_rounded_to_tick=price_rounded,
                )
            delta = delta_q

            if abs(delta) * fill_price < self.constraints.min_notional_usdt:
                self.rejected_min_notional += 1
                return ExecutionResult(
                    attempted=True,
                    filled=False,
                    reject_reason="min_notional",
                    qty_rounded_to_step=qty_rounded,
                    price_rounded_to_tick=price_rounded,
                )

        # For sells, ensure we still don't exceed pos after rounding.
        if delta < 0:
            delta = max(delta, -self.pos_qty)

        # For buys, after rounding, re-check affordability (should be safe due to floor)
        if delta > 0:
            denom = float(fill_price) * (1.0 + float(fee_rate))
            if denom > 0:
                max_buy_qty = float(self.cash) / denom
                if delta > max_buy_qty + 1e-12:
                    delta = max_buy_qty
                    buy_clamped = True

        fee = self._apply_fill(
            dt=str(dt),
            delta_qty=float(delta),
            fill_price=float(fill_price),
            fee_rate=float(fee_rate),
            order_type=str(order_type.value),
            slip_bps=float(applied_slip_bps),
            spread_bps=float(applied_spread_bps),
            liq_mult=float(liq_mult),
        )

        return ExecutionResult(
            attempted=True,
            filled=True,
            reject_reason=None,
            leverage_clamped=bool(buy_clamped),
            qty_rounded_to_step=qty_rounded,
            price_rounded_to_tick=price_rounded,
            delta_qty=float(delta),
            fill_price=float(fill_price),
            fee_paid=float(fee),
        )

    def mark(self, dt: str, mark_price: float):
        if self.record_equity_curve:
            eq = self.equity(mark_price)
            self.equity_curve.append(
                {
                    "dt": str(dt),
                    "price": float(mark_price),
                    "equity": float(eq),
                    "cashflow": float(getattr(self, "cashflow_this_bar", 0.0) or 0.0),
                    "cash": float(self.cash),
                    "pos_qty": float(self.pos_qty),
                    "avg_entry": float(self.avg_entry),
                    "realized_pnl_gross": float(self.realized_pnl),
                    "unrealized_pnl": float(self.unrealized_pnl(mark_price)),
                    "fees_paid_total": float(self.fees_paid_total),
                    "fees_paid_taker": float(self.fees_paid_taker),
                    "fees_paid_maker": float(self.fees_paid_maker),
                    "funding_net": float(self.funding_net),
                    "funding_paid": float(self.funding_paid),
                    "funding_received": float(self.funding_received),
                    "funding_penalty": float(self.funding_penalty),
                    "rej_min_notional": int(self.rejected_min_notional),
                    "rej_qty_step": int(self.rejected_qty_step),
                    "price_tick_rounds": int(self.price_tick_rounds),
                }
            )

# ============================================================
# Broker (pure accounting + exchange constraints)
# ============================================================


class PaperPerpsBroker:
    def __init__(
        self, starting_equity: float, constraints: EngineConstraints, cfg: BacktestConfig
    ):
        self.record_fills = True
        self.record_equity_curve = True
        self.cfg = cfg
        self.constraints = constraints

        self.cash = float(starting_equity)
        self.pos_qty = 0.0
        self.avg_entry = 0.0

        self.realized_pnl = 0.0

        self.fees_paid_total = 0.0
        self.fees_paid_taker = 0.0
        self.fees_paid_maker = 0.0

        self.funding_net = 0.0
        self.funding_paid = 0.0
        self.funding_received = 0.0
        self.funding_penalty = 0.0

        self.rejected_min_notional = 0
        self.rejected_qty_step = 0
        self.price_tick_rounds = 0

        self.fills: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []

    def set_recording(self, *, record_fills: bool, record_equity_curve: bool) -> None:
        self.record_fills = bool(record_fills)
        self.record_equity_curve = bool(record_equity_curve)    

    def unrealized_pnl(self, mark_price: float) -> float:
        return (
            0.0
            if abs(self.pos_qty) < 1e-12
            else (mark_price - self.avg_entry) * self.pos_qty
        )

    def equity(self, mark_price: float) -> float:
        return self.cash + self.unrealized_pnl(mark_price)

    def _cap_delta_for_leverage(self, delta_qty: float, price: float) -> Tuple[float, bool]:
        if price <= 0:
            return 0.0, False

        eq_before = self.equity(price)
        max_notional = max(0.0, eq_before * float(self.constraints.max_leverage))

        desired_new_qty = self.pos_qty + delta_qty
        desired_notional = abs(desired_new_qty) * price

        if desired_notional <= max_notional:
            return delta_qty, False

        current_notional = abs(self.pos_qty) * price
        if current_notional > max_notional and abs(desired_new_qty) < abs(self.pos_qty):
            return delta_qty, False

        sign = 1.0 if desired_new_qty > 0 else (-1.0 if desired_new_qty < 0 else 0.0)
        capped_new_qty = sign * (max_notional / price) if sign != 0.0 else 0.0
        return capped_new_qty - self.pos_qty, True

    def _apply_fill(
        self,
        dt: str,
        delta_qty: float,
        fill_price: float,
        fee_rate: float,
        order_type: str,
        slip_bps: float,
        spread_bps: float,
        liq_mult: float,
    ):
        if abs(delta_qty) < 1e-12:
            return 0.0

        old_qty = self.pos_qty
        new_qty = old_qty + delta_qty

        fee = abs(delta_qty) * fill_price * fee_rate
        self.cash -= fee

        self.fees_paid_total += fee
        if order_type == "market":
            self.fees_paid_taker += fee
        else:
            self.fees_paid_maker += fee

        if abs(old_qty) < 1e-12:
            self.pos_qty = new_qty
            self.avg_entry = fill_price if abs(new_qty) > 1e-12 else 0.0
        elif (old_qty > 0 and new_qty > 0 and delta_qty > 0) or (
            old_qty < 0 and new_qty < 0 and delta_qty < 0
        ):
            total_abs = abs(old_qty) + abs(delta_qty)
            self.avg_entry = (
                abs(old_qty) * self.avg_entry + abs(delta_qty) * fill_price
            ) / total_abs
            self.pos_qty = new_qty
        else:
            closed_qty = min(abs(delta_qty), abs(old_qty))
            sign_old = 1.0 if old_qty > 0 else -1.0
            realized = (fill_price - self.avg_entry) * closed_qty * sign_old

            self.realized_pnl += realized
            self.cash += realized

            if abs(new_qty) < 1e-12:
                self.pos_qty = 0.0
                self.avg_entry = 0.0
            else:
                self.pos_qty = new_qty
                if (old_qty > 0 > new_qty) or (old_qty < 0 < new_qty):
                    self.avg_entry = fill_price

        side = "buy" if delta_qty > 0 else "sell"
        if self.record_fills:
            self.fills.append(
                {
                    "dt": str(dt),
                    "side": side,
                    "qty": float(delta_qty),
                    "price": float(fill_price),
                    "fee": float(fee),
                    "fee_rate": float(fee_rate),
                    "order_type": str(order_type),
                    "slip_bps": float(slip_bps),
                    "spread_bps": float(spread_bps),
                    "liq_mult": float(liq_mult),
                    "pos_qty": float(self.pos_qty),
                    "avg_entry": float(self.avg_entry),
                    "realized_pnl": float(self.realized_pnl),
                    "equity": float(self.equity(fill_price)),
                }
            )

        return fee

    def trade_to_target_qty(
        self,
        dt: str,
        target_qty: float,
        ref_price: float,
        order_type: OrderType,
        slip_bps: float,
        spread_bps: float,
        liq_mult: float,
        force_close: bool = False,
    ) -> ExecutionResult:
        if ref_price <= 0:
            return ExecutionResult(attempted=True, filled=False, reject_reason="bad_price")

        delta = (-self.pos_qty) if (force_close and abs(target_qty) < 1e-12) else (
            target_qty - self.pos_qty
        )
        if abs(delta) < 1e-12:
            return ExecutionResult(attempted=True, filled=False, reject_reason="no_change")

        fee_rate = self.cfg.fee_taker if order_type == OrderType.MARKET else self.cfg.fee_maker
        side_sign = 1 if delta > 0 else -1

        applied_spread_bps = 0.0
        applied_slip_bps = 0.0
        price_rounded = False

        if order_type == OrderType.LIMIT:
            pre = ref_price
            ticked, price_rounded = round_price_to_tick(
                pre, self.constraints.price_tick, side_sign
            )
            if price_rounded:
                self.price_tick_rounds += 1
            fill_price = ticked
        else:
            applied_spread_bps = float(spread_bps)
            applied_slip_bps = float(slip_bps)

            half_spread = 0.5 * bps_to_frac(applied_spread_bps)
            spread_price = (
                ref_price * (1.0 + half_spread)
                if side_sign > 0
                else ref_price * (1.0 - half_spread)
            )

            slip = bps_to_frac(applied_slip_bps)
            pre = (
                spread_price * (1.0 + slip)
                if side_sign > 0
                else spread_price * (1.0 - slip)
            )

            ticked, price_rounded = round_price_to_tick(
                pre, self.constraints.price_tick, side_sign
            )
            if price_rounded:
                self.price_tick_rounds += 1
            fill_price = ticked

        delta, lev_clamped = self._cap_delta_for_leverage(delta, fill_price)

        qty_rounded = False
        if not (force_close and abs(target_qty) < 1e-12):
            delta_q, qty_rounded = quantize_delta_to_step(delta, self.constraints.qty_step)
            if abs(delta_q) < 1e-12:
                self.rejected_qty_step += 1
                return ExecutionResult(
                    attempted=True,
                    filled=False,
                    reject_reason="qty_step",
                    leverage_clamped=lev_clamped,
                    qty_rounded_to_step=qty_rounded,
                    price_rounded_to_tick=price_rounded,
                )
            delta = delta_q

            if abs(delta) * fill_price < self.constraints.min_notional_usdt:
                self.rejected_min_notional += 1
                return ExecutionResult(
                    attempted=True,
                    filled=False,
                    reject_reason="min_notional",
                    leverage_clamped=lev_clamped,
                    qty_rounded_to_step=qty_rounded,
                    price_rounded_to_tick=price_rounded,
                )

        fee = self._apply_fill(
            dt=str(dt),
            delta_qty=delta,
            fill_price=float(fill_price),
            fee_rate=float(fee_rate),
            order_type=str(order_type.value),
            slip_bps=float(applied_slip_bps),
            spread_bps=float(applied_spread_bps),
            liq_mult=float(liq_mult),
        )

        return ExecutionResult(
            attempted=True,
            filled=True,
            reject_reason=None,
            leverage_clamped=lev_clamped,
            qty_rounded_to_step=qty_rounded,
            price_rounded_to_tick=price_rounded,
            delta_qty=float(delta),
            fill_price=float(fill_price),
            fee_paid=float(fee),
        )

    def mark(self, dt: str, mark_price: float):
        if self.record_equity_curve:
            eq = self.equity(mark_price)
            self.equity_curve.append(
                {
                    "dt": str(dt),
                    "price": float(mark_price),
                    "equity": float(eq),
                    "cashflow": float(getattr(self, "cashflow_this_bar", 0.0) or 0.0),
                    "cash": float(self.cash),
                    "pos_qty": float(self.pos_qty),
                    "avg_entry": float(self.avg_entry),
                    "realized_pnl_gross": float(self.realized_pnl),
                    "unrealized_pnl": float(self.unrealized_pnl(mark_price)),
                    "fees_paid_total": float(self.fees_paid_total),
                    "fees_paid_taker": float(self.fees_paid_taker),
                    "fees_paid_maker": float(self.fees_paid_maker),
                    "funding_net": float(self.funding_net),
                    "funding_paid": float(self.funding_paid),
                    "funding_received": float(self.funding_received),
                    "funding_penalty": float(self.funding_penalty),
                    "rej_min_notional": int(self.rejected_min_notional),
                    "rej_qty_step": int(self.rejected_qty_step),
                    "price_tick_rounds": int(self.price_tick_rounds),
                }
            )


# ============================================================
# Engine realism models
# ============================================================


def slip_bps_for(cfg: BacktestConfig, order_type: OrderType, vol_bps: float, liq_mult: float) -> float:
    if order_type == OrderType.MARKET:
        base = cfg.slippage_taker_bps + cfg.slippage_taker_vol_mult * vol_bps
        if cfg.liq_apply_to_slippage:
            base *= max(1.0, float(liq_mult))
        return clamp(base, 0.0, cfg.slippage_taker_bps_max)

    base = cfg.slippage_maker_bps + cfg.slippage_maker_vol_mult * vol_bps
    if cfg.liq_apply_to_slippage:
        base *= max(1.0, float(liq_mult))
    return clamp(base, 0.0, cfg.slippage_maker_bps_max)


def spread_bps_for(cfg: BacktestConfig, vol_bps: float, liq_mult: float) -> float:
    base = cfg.spread_bps + cfg.spread_vol_mult * vol_bps
    if cfg.liq_apply_to_spread:
        base *= max(1.0, float(liq_mult))
    return clamp(base, 0.0, cfg.spread_bps_max)


def tp_limit_fill_prob(cfg: BacktestConfig, vol_bps: float, overshoot_bps: float) -> float:
    p = (
        cfg.tp_limit_fill_prob_base
        - cfg.tp_limit_fill_vol_penalty * vol_bps
        + cfg.tp_limit_fill_overshoot_bonus * overshoot_bps
    )
    return clamp(p, cfg.tp_limit_fill_prob_min, 1.0)


# ============================================================
# Backtest core (plan-driven)
# ============================================================


def run_backtest_once(
    df: pd.DataFrame,
    strategy,  # must implement: on_bar(ctx: StrategyContext) -> PlanUpdate
    seed: int,
    starting_equity: float,
    constraints: EngineConstraints,
    cfg: BacktestConfig,
    verbose: bool = False,
    show_progress: bool = False,
    features_ready: bool = False,
    record_fills: bool = True,
    record_equity_curve: bool = True,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    rng = random.Random(int(seed))

    is_spot = str(getattr(cfg, "market_mode", "spot")).lower() == "spot"
    # --- PATCH: FEATURE STORE INTEGRATION ---
    # 1. Enrich data with indicators (skip if already done by batch runner)
    if not features_ready:
        df = add_features(df)

    # NOTE: add_features() must not forward-fill event columns like funding_rate.
    # 2. Pre-extract feature columns for fast lookup inside the loop
    reserved_cols = {
        "dt",
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vol_bps",
        "liq_mult",
        # Perp realism columns (not strategy features):
        "mark_open",
        "mark_high",
        "mark_low",
        "mark_close",
        "index_open",
        "index_high",
        "index_low",
        "index_close",
        "funding_rate",
        "is_funding_event",
    }
    feature_cols = [c for c in df.columns if c not in reserved_cols]

    # Create a dict of numpy arrays (faster than DF indexing row by row)
    feature_arrays = {c: df[c].to_numpy(dtype=np.float64) for c in feature_cols}
    feature_view = FeatureView(feature_arrays)
    # ----------------------------------------

    required = {"dt", "open", "high", "low", "close", "vol_bps"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")

    has_mark = all(
        c in df.columns for c in ("mark_open", "mark_high", "mark_low", "mark_close")
    )
    if (not is_spot) and (not has_mark):
        warnings.warn(
            "mark_* columns not found; falling back to last OHLC for mark-to-market, "
            "stops, liquidation. This is not perp-realistic."
        )

    has_funding = "funding_rate" in df.columns
    if (not is_spot) and cfg.use_real_funding and (not has_funding):
        warnings.warn(
            "cfg.use_real_funding=True but funding_rate column not found; falling "
            "back to synthetic funding model."
        )

    has_ts = "ts" in df.columns
    has_volume = "volume" in df.columns
    if "liq_mult" not in df.columns:
        df = df.copy()
        df["liq_mult"] = 1.0

    dts = df["dt"].astype(str).to_numpy()
    opens = df["open"].to_numpy(dtype=np.float64)
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    vol_bps_arr = df["vol_bps"].to_numpy(dtype=np.float64)
    liq_mult_arr = df["liq_mult"].to_numpy(dtype=np.float64)
    if has_mark:
        mark_opens = df["mark_open"].to_numpy(dtype=np.float64)
        mark_highs = df["mark_high"].to_numpy(dtype=np.float64)
        mark_lows = df["mark_low"].to_numpy(dtype=np.float64)
        mark_closes = df["mark_close"].to_numpy(dtype=np.float64)
    else:
        mark_opens = opens
        mark_highs = highs
        mark_lows = lows
        mark_closes = closes

    funding_rate_arr = (
        df["funding_rate"].to_numpy(dtype=np.float64) if has_funding else None
    )
    vols = df["volume"].to_numpy(dtype=np.float64) if has_volume else None
    ts = df["ts"].to_numpy(dtype=np.int64) if has_ts else None
    dt_dates = pd.to_datetime(df["dt"], utc=True).dt.date.to_numpy()

    # PRECOMPUTE dt as datetime64[ns] once (avoid per-bar pd.to_datetime calls)
    # ts is in milliseconds in this project (see _normalize_columns).
    if ts is not None:
        dt_ns_arr = (ts.astype(np.int64) * 1_000_000).astype("datetime64[ns]")
    else:
        dt_ns_arr = (
            pd.to_datetime(df["dt"], utc=True)
            .dt.tz_localize(None)
            .to_numpy(dtype="datetime64[ns]")
        )

    if is_spot:
        broker = PaperSpotBroker(
            starting_cash=float(starting_equity),
            constraints=constraints,
            cfg=cfg,
        )
    else:
        broker = PaperPerpsBroker(
            starting_equity=float(starting_equity),
            constraints=constraints,
            cfg=cfg,
        )
    broker.set_recording(record_fills=record_fills, record_equity_curve=record_equity_curve)

    active_plan: Optional[TradePlan] = None
    pending_update: Optional[PlanUpdate] = None
    pending_bars_left: Optional[int] = None

    last_exec: Optional[ExecutionResult] = None

    trades: List[Dict[str, Any]] = []
    baseline_qty_abs = 0.0
    active_stop: Optional[float] = None
    tps: List[TakeProfitLevel] = []
    tps_hit = 0

    cooldown_until: Optional[np.datetime64] = None
    loss_streak = 0
    max_loss_streak = 0
    cooldown_triggers = 0

    blocked_days = set()
    daily_loss_triggers = 0
    daily_blocked_date = None
    current_day = None
    day_start_equity = None

    fund_bucket = None
    fund_step = cfg.funding_interval_hours * 3600
    fund_jitter_fracs: List[float] = []
    funding_events_real = 0

    tp_limit_attempts = 0
    tp_limit_fills = 0
    tp_limit_misses = 0

    liquidated = False
    cashflow_this_bar = 0.0

    def account_state(mark_px: float) -> AccountState:
        return AccountState(
            cash=float(broker.cash),
            equity=float(broker.equity(mark_px)),
            realized_pnl_gross=float(broker.realized_pnl),
            fees_paid_total=float(broker.fees_paid_total),
            funding_net=float(broker.funding_net),
        )

    def position_state(mark_px: float) -> PositionState:
        liq = None
        if not is_spot:
            liq = liquidation_price(
                broker.cash, broker.avg_entry, broker.pos_qty, constraints.maint_margin_rate
            )
        return PositionState(
            qty=float(broker.pos_qty),
            avg_entry=float(broker.avg_entry),
            unrealized_pnl=float(broker.unrealized_pnl(mark_px)),
            liq_price=float(liq) if liq is not None else None,
        )

    def maybe_apply_funding(i: int, mark_price: float):
        nonlocal funding_events_real
        if is_spot:
            return

        # Prefer real funding if present.
        if cfg.use_real_funding and funding_rate_arr is not None:
            rate = float(funding_rate_arr[i])
            if (not math.isfinite(rate)) or abs(rate) <= 0.0:
                return
            if abs(broker.pos_qty) < 1e-12:
                funding_events_real += 1
                return

            notional = abs(broker.pos_qty) * float(mark_price)
            # Binance-style: positive rate => longs pay shorts
            f_total = -sgn(broker.pos_qty) * notional * rate

            broker.cash += f_total
            broker.funding_net += f_total

            if f_total < 0:
                broker.funding_paid += -f_total
            else:
                broker.funding_received += f_total

            funding_events_real += 1
            return

        nonlocal fund_bucket
        if cfg.funding_rate_8h == 0 or ts is None:
            return
        ts_sec = int(ts[i] // 1000)
        b = ts_sec // fund_step
        if fund_bucket is None:
            fund_bucket = b
            return
        if b == fund_bucket:
            return
        fund_bucket = b
        if abs(broker.pos_qty) < 1e-12:
            return

        notional = abs(broker.pos_qty) * mark_price
        f_base = -sgn(broker.pos_qty) * notional * cfg.funding_rate_8h

        jitter_frac = rng.random() * cfg.funding_jitter_max_frac
        f_penalty = -abs(notional * cfg.funding_rate_8h) * jitter_frac
        fund_jitter_fracs.append(jitter_frac)

        f_total = f_base + f_penalty

        broker.cash += f_total
        broker.funding_net += f_total
        broker.funding_penalty += f_penalty

        if f_total < 0:
            broker.funding_paid += -f_total
        else:
            broker.funding_received += f_total

    def start_trade_record(
        entry_dt: str,
        start_realized_gross: Optional[float] = None,
        start_fees_total: Optional[float] = None,
        start_funding_net: Optional[float] = None,
    ):
        return {
            "entry_dt": entry_dt,
            "exit_dt": None,
            "exit_reason": None,
            "side": None,
            "entry_price": None,
            "qty_initial": None,
            "notional_entry": None,
            # --- TP ladder telemetry ---
            "tp_levels": 0,
            "tp_prices": [],
            "tps_hit": 0,
            # --- Stop telemetry ---
            "stop_init": None,
            "stop_last": None,
            "stop_moved": False,
            "stop_move_count": 0,
            "stop_move_tp_count": 0,
            "stop_move_plan_count": 0,
            "stop_hit_price": None,
            "stop_hit_was_moved": None,
            # --------------------------
            "risk_usd": None,
            "leverage_target": None,
            "alloc_equity": None,
            "plan_meta": {},
            "start_realized_gross": float(broker.realized_pnl if start_realized_gross is None else start_realized_gross),
            "start_fees_total": float(broker.fees_paid_total if start_fees_total is None else start_fees_total),
            "start_funding_net": float(broker.funding_net if start_funding_net is None else start_funding_net),
            "end_realized_gross": None,
            "end_fees_total": None,
            "end_funding_net": None,
            "gross_pnl": None,
            "fees_total": None,
            "funding_net": None,
            "net_pnl": None,
            "net_R": None,
            "fee_R": None,
        }

    def finalize_trade(exit_dt: str, reason: str):
        if not trades or trades[-1]["exit_dt"] is not None:
            return
        tr = trades[-1]
        tr["exit_dt"] = str(exit_dt)
        tr["exit_reason"] = str(reason)

        tr["end_realized_gross"] = float(broker.realized_pnl)
        tr["end_fees_total"] = float(broker.fees_paid_total)
        tr["end_funding_net"] = float(broker.funding_net)

        tr["gross_pnl"] = float(tr["end_realized_gross"] - tr["start_realized_gross"])
        fees_total = float(tr["end_fees_total"] - tr["start_fees_total"])
        funding_net = float(tr["end_funding_net"] - tr["start_funding_net"])
        tr["fees_total"] = fees_total
        tr["funding_net"] = funding_net
        tr["net_pnl"] = float(tr["gross_pnl"] + funding_net - fees_total)

        risk_usd = tr.get("risk_usd")
        if risk_usd and risk_usd > 0:
            tr["net_R"] = float(tr["net_pnl"] / risk_usd)
            tr["fee_R"] = float(fees_total / risk_usd)

    def apply_cooldowns_after_trade(exit_dt_ns: np.datetime64, net_pnl: float, guard):
        nonlocal cooldown_until, loss_streak, max_loss_streak, cooldown_triggers

        if guard is None:
            return

        if guard.base_cooldown_minutes and guard.base_cooldown_minutes > 0:
            base_until = exit_dt_ns + np.timedelta64(int(guard.base_cooldown_minutes), "m")
            cooldown_until = base_until if cooldown_until is None else max(cooldown_until, base_until)

        if net_pnl < 0:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0

        if guard.loss_streak_start and guard.loss_streak_start > 0 and loss_streak >= guard.loss_streak_start:
            k = loss_streak - guard.loss_streak_start
            mins = min(int(guard.cooldown_base_min) * (2**k), int(guard.cooldown_max_min))
            exp_until = exit_dt_ns + np.timedelta64(int(mins), "m")
            cooldown_until = exp_until if cooldown_until is None else max(cooldown_until, exp_until)
            cooldown_triggers += 1

    def build_target_qty(plan: TradePlan, ref_px: float) -> float:
        if plan.target_qty is not None:
            return float(plan.target_qty)
        if plan.target_notional is not None:
            if ref_px <= 0:
                return 0.0
            return float(plan.desired_side) * (float(plan.target_notional) / float(ref_px))
        return 0.0

    def apply_plan_at_open(dt: str, o: float, h: float, l: float, vol_bps: float, liq_mult: float) -> ExecutionResult:
        nonlocal active_plan, active_stop, tps, tps_hit, baseline_qty_abs, pending_bars_left, cashflow_this_bar

        if pending_update is None:
            return ExecutionResult(attempted=False, filled=False)

        upd = pending_update

        if active_plan is not None and pending_bars_left is not None:
            pending_bars_left = max(0, pending_bars_left - 1)
            if pending_bars_left == 0:
                active_plan = None
                active_stop = None
                tps = []
                tps_hit = 0

        if upd.action == PlanAction.HOLD:
            return ExecutionResult(attempted=False, filled=False)

        if upd.action == PlanAction.CANCEL:
            active_plan = None
            active_stop = None
            tps = []
            tps_hit = 0
            pending_bars_left = None
            return ExecutionResult(attempted=True, filled=False, reject_reason="plan_cancelled")

        if upd.action == PlanAction.REPLACE:
            if upd.plan is None:
                return ExecutionResult(attempted=True, filled=False, reject_reason="replace_without_plan")

            prev_stop_before_replace = active_stop

            active_plan = upd.plan
            pending_bars_left = active_plan.expires_after_bars

            # Apply external cashflow at bar open (deposit/withdraw), independent of trade fill.
            cd = float(getattr(active_plan, "cash_delta", 0.0) or 0.0)
            if abs(cd) > 1e-12:
                broker.cash += cd
                cashflow_this_bar += cd

            active_stop = float(active_plan.stop_price) if active_plan.stop_price is not None else None
            tps = list(active_plan.take_profits or [])
            tps_hit = 0

            target_qty = build_target_qty(active_plan, o)

            pos_before = broker.pos_qty

            # PRE-EXEC snapshots
            real_before = float(broker.realized_pnl)
            fees_before = float(broker.fees_paid_total)
            fund_before = float(broker.funding_net)

            if active_plan.entry_order_type == OrderType.LIMIT:
                if active_plan.entry_limit_price is None:
                    return ExecutionResult(attempted=True, filled=False, reject_reason="limit_missing_price")
                limit_px = float(active_plan.entry_limit_price)

                delta = target_qty - broker.pos_qty
                if abs(delta) < 1e-12:
                    return ExecutionResult(attempted=True, filled=False, reject_reason="no_change")

                touched = (l <= limit_px) if delta > 0 else (h >= limit_px)
                if not touched:
                    return ExecutionResult(attempted=True, filled=False, reject_reason="limit_not_touched")

                exec_res = broker.trade_to_target_qty(
                    dt=dt,
                    target_qty=float(target_qty),
                    ref_price=float(limit_px),
                    order_type=OrderType.LIMIT,
                    slip_bps=slip_bps_for(cfg, OrderType.LIMIT, vol_bps, liq_mult),
                    spread_bps=spread_bps_for(cfg, vol_bps, liq_mult),
                    liq_mult=liq_mult,
                    force_close=(abs(target_qty) < 1e-12),
                )
            else:
                exec_res = broker.trade_to_target_qty(
                    dt=dt,
                    target_qty=float(target_qty),
                    ref_price=float(o),
                    order_type=OrderType.MARKET,
                    slip_bps=slip_bps_for(cfg, OrderType.MARKET, vol_bps, liq_mult),
                    spread_bps=spread_bps_for(cfg, vol_bps, liq_mult),
                    liq_mult=liq_mult,
                    force_close=(abs(target_qty) < 1e-12),
                )

            pos_after = broker.pos_qty
            opened = (abs(pos_before) < 1e-12) and (abs(pos_after) >= 1e-12)
            closed = (abs(pos_before) >= 1e-12) and (abs(pos_after) < 1e-12)

            if opened:
                baseline_qty_abs = abs(broker.pos_qty)

                tr = start_trade_record(
                    entry_dt=dt,
                    start_realized_gross=real_before,
                    start_fees_total=fees_before,
                    start_funding_net=fund_before,
                )
                tr["side"] = 1 if broker.pos_qty > 0 else -1
                tr["entry_price"] = float(broker.avg_entry)
                tr["qty_initial"] = float(broker.pos_qty)
                tr["notional_entry"] = float(abs(broker.pos_qty) * broker.avg_entry)

                tr["tp_levels"] = int(len(tps))
                tr["tp_prices"] = [float(x.price) for x in tps]

                tr["stop_init"] = float(active_stop) if active_stop is not None else None
                tr["stop_last"] = tr["stop_init"]

                meta = dict(active_plan.metadata or {})
                tr["plan_meta"] = meta
                tr["risk_usd"] = meta.get("risk_usd", None)

                # Auto-calculate risk if stop exists but risk_usd not provided
                if tr["risk_usd"] is None and tr["stop_init"] is not None and tr["entry_price"] > 0:
                    dist = abs(tr["entry_price"] - tr["stop_init"])
                    tr["risk_usd"] = dist * abs(tr["qty_initial"])

                tr["leverage_target"] = meta.get("leverage_target", None)
                tr["alloc_equity"] = meta.get("alloc_equity", None)

                trades.append(tr)

            # Plan replace while in-trade
            still_in_trade = (abs(pos_before) >= 1e-12) and (abs(pos_after) >= 1e-12)
            trade_open = trades and trades[-1]["exit_dt"] is None
            if still_in_trade and trade_open:
                tr = trades[-1]
                old_stop = tr.get("stop_last", tr.get("stop_init", None))
                new_stop = float(active_stop) if active_stop is not None else None

                if new_stop is not None:
                    changed = (old_stop is None) or (abs(float(new_stop) - float(old_stop)) > 1e-9)
                    if changed:
                        tr["stop_moved"] = True
                        tr["stop_move_count"] = int(tr.get("stop_move_count", 0)) + 1
                        tr["stop_move_plan_count"] = int(tr.get("stop_move_plan_count", 0)) + 1
                        tr["stop_last"] = float(new_stop)

                _ = prev_stop_before_replace

            if closed:
                baseline_qty_abs = 0.0
                tps_hit = 0
                finalize_trade(exit_dt=dt, reason="plan_flat")

                guard2 = active_plan.guardrails if active_plan is not None else None
                if trades and trades[-1]["exit_dt"] is not None:
                    apply_cooldowns_after_trade(_dt64_utc_naive(dt), float(trades[-1]["net_pnl"]), guard2)

            return exec_res

        return ExecutionResult(attempted=False, filled=False)

    # Progress
    n = len(df)
    it = range(n)
    use_tqdm = False
    if show_progress:
        try:
            from tqdm import tqdm

            it = tqdm(it, total=n, desc="Backtest", unit="bar", smoothing=0.05)
            use_tqdm = True
        except Exception:
            use_tqdm = False

    last_pct_printed = -1

    for i in it:
        if show_progress and (not use_tqdm) and n > 0:
            pct = int((i + 1) * 100 / n)
            if pct % 5 == 0 and pct != last_pct_printed:
                last_pct_printed = pct
                sys.stdout.write(f"\rBacktest progress: {pct}% ({i+1}/{n})")
                sys.stdout.flush()

        dt = str(dts[i])
        o = float(opens[i])
        h = float(highs[i])
        l = float(lows[i])
        c = float(closes[i])
        mo = float(mark_opens[i])
        mh = float(mark_highs[i])
        ml = float(mark_lows[i])
        mc = float(mark_closes[i])
        vol_bps = float(vol_bps_arr[i])
        liq_mult = float(liq_mult_arr[i])
        v = float(vols[i]) if vols is not None else None

        cashflow_this_bar = 0.0
        broker.cashflow_this_bar = 0.0

        dt_day = dt_dates[i]
        if current_day != dt_day:
            current_day = dt_day
            daily_blocked_date = None
            day_start_equity = None

        guard = active_plan.guardrails if (active_plan and active_plan.guardrails) else None
        dt_ns = dt_ns_arr[i]

        blocked = (daily_blocked_date == current_day) if daily_blocked_date is not None else False
        in_cooldown = (cooldown_until is not None and dt_ns < cooldown_until)

        can_apply = True
        if guard is not None:
            can_apply = (not blocked) and (not in_cooldown)

        # Funding happens at known timestamps; apply at bar open-time using mark open.
        # This ensures positions held from the previous minute pay/receive, and a new
        # entry on the funding minute does not retroactively pay funding.
        maybe_apply_funding(i, mo)
        if pending_update is not None and can_apply:
            last_exec = apply_plan_at_open(dt, o, h, l, vol_bps, liq_mult)
        else:
            last_exec = (
                ExecutionResult(attempted=False, filled=False, reject_reason="blocked_or_cooldown")
                if pending_update is not None
                else last_exec
            )

        if day_start_equity is None:
            day_start_equity = broker.equity(mo)

        in_trade = abs(broker.pos_qty) >= 1e-12
        if in_trade and trades and trades[-1]["exit_dt"] is None:
            side = 1 if broker.pos_qty > 0 else -1
            
            liq_hit = False
            liq_ref = float(mo)
            if not is_spot:
                liq_px = None
            if not is_spot:
                liq_px = liquidation_price(
                    broker.cash,
                    broker.avg_entry,
                    broker.pos_qty,
                    constraints.maint_margin_rate,
                )
                if liq_px is not None:
                    buf = float(cfg.liq_brutal_buffer_frac)
                    if side > 0:
                        thresh = float(liq_px) * (1.0 + buf)
                        liq_hit = (mo <= thresh) or (ml <= thresh)
                        liq_ref = min(mo, thresh)
                    else:
                        thresh = float(liq_px) * (1.0 - buf)
                        liq_hit = (mo >= thresh) or (mh >= thresh)
                        liq_ref = max(mo, thresh)

                if liq_hit:
                    liquidated = True
                    # Accounting fix:
                    # We terminate the account at equity=0. Capture the equity-at-liq
                    # as a realized loss so trade PnL reconciles with the equity curve.
                    eq_before = float(broker.equity(float(liq_ref)))
                    liq_loss = max(0.0, eq_before)
                    broker.realized_pnl -= liq_loss

                    finalize_trade(exit_dt=dt, reason="liquidation")
                    broker.cash = 0.0
                    broker.pos_qty = 0.0
                    broker.avg_entry = 0.0
                    broker.mark(dt, float(liq_ref))
                    break

            if active_stop is not None:
                if side > 0:
                    hit, ref_px = long_stop_hit_mark(
                        mark_o=mo,
                        mark_l=ml,
                        stop_price=float(active_stop),
                        last_o=o,
                        last_l=l,
                    )
                else:
                    hit, ref_px = short_stop_hit_mark(
                        mark_o=mo,
                        mark_h=mh,
                        stop_price=float(active_stop),
                        last_o=o,
                        last_h=h,
                    )

                if hit:
                    broker.trade_to_target_qty(
                        dt=dt,
                        target_qty=0.0,
                        ref_price=float(ref_px),
                        order_type=OrderType.MARKET,
                        slip_bps=slip_bps_for(cfg, OrderType.MARKET, vol_bps, liq_mult),
                        spread_bps=spread_bps_for(cfg, vol_bps, liq_mult),
                        liq_mult=liq_mult,
                        force_close=True,
                    )

                    if trades and trades[-1]["exit_dt"] is None:
                        tr = trades[-1]
                        tr["stop_hit_price"] = float(ref_px)

                        s0 = tr.get("stop_init", None)
                        sN = float(active_stop) if active_stop is not None else None
                        moved = False
                        if s0 is not None and sN is not None:
                            moved = abs(float(sN) - float(s0)) > 1e-9
                        else:
                            moved = bool(tr.get("stop_moved", False))

                        tr["stop_hit_was_moved"] = bool(moved)
                        tr["stop_last"] = sN

                    finalize_trade(exit_dt=dt, reason="stop")
                    baseline_qty_abs = 0.0
                    tps_hit = 0

                    guard2 = active_plan.guardrails if (active_plan and active_plan.guardrails) else None
                    if trades and trades[-1]["exit_dt"] is not None:
                        apply_cooldowns_after_trade(dt_ns, float(trades[-1]["net_pnl"]), guard2)

            in_trade = abs(broker.pos_qty) >= 1e-12
            if in_trade and trades and trades[-1]["exit_dt"] is None and tps:
                reached = tps_hit - 1
                while reached + 1 < len(tps):
                    nxt = reached + 1
                    tp_px = float(tps[nxt].price)
                    if (side > 0 and h >= tp_px) or (side < 0 and l <= tp_px):
                        reached = nxt
                        continue
                    break

                for tp_i in range(tps_hit, reached + 1):
                    level = tps[tp_i]
                    tp_px = float(level.price)

                    if level.order_type == OrderType.LIMIT:
                        tp_limit_attempts += 1
                        if cfg.force_tp_limit_fills:
                            tp_limit_fills += 1
                        else:
                            if side > 0:
                                overshoot_bps = max(0.0, (h - tp_px) / tp_px) * 10_000.0
                            else:
                                overshoot_bps = max(0.0, (tp_px - l) / tp_px) * 10_000.0
                            p_fill = tp_limit_fill_prob(cfg, vol_bps, overshoot_bps)
                            if rng.random() > p_fill:
                                tp_limit_misses += 1
                                break
                            tp_limit_fills += 1

                    if baseline_qty_abs <= 1e-12:
                        baseline_qty_abs = abs(broker.pos_qty)

                    close_qty_abs = baseline_qty_abs * float(level.fraction_of_initial)
                    close_qty_abs = min(close_qty_abs, abs(broker.pos_qty))

                    if tp_i == len(tps) - 1:
                        close_qty_abs = abs(broker.pos_qty)

                    if close_qty_abs <= 1e-12:
                        continue

                    target_qty = float(broker.pos_qty - side * close_qty_abs)

                    broker.trade_to_target_qty(
                        dt=dt,
                        target_qty=float(target_qty),
                        ref_price=float(tp_px),
                        order_type=level.order_type,
                        slip_bps=slip_bps_for(cfg, level.order_type, vol_bps, liq_mult),
                        spread_bps=spread_bps_for(cfg, vol_bps, liq_mult),
                        liq_mult=liq_mult,
                        force_close=(tp_i == len(tps) - 1 or abs(target_qty) < 1e-12),
                    )

                    tps_hit = tp_i + 1
                    if trades and trades[-1]["exit_dt"] is None:
                        trades[-1]["tps_hit"] = int(tps_hit)

                    if level.move_stop_to is not None:
                        new_stop = float(level.move_stop_to)
                        old_stop = float(active_stop) if active_stop is not None else None
                        active_stop = new_stop

                        if trades and trades[-1]["exit_dt"] is None:
                            tr = trades[-1]
                            changed = (old_stop is None) or (abs(float(new_stop) - float(old_stop)) > 1e-9)
                            if changed:
                                tr["stop_moved"] = True
                                tr["stop_move_count"] = int(tr.get("stop_move_count", 0)) + 1
                                tr["stop_move_tp_count"] = int(tr.get("stop_move_tp_count", 0)) + 1
                                tr["stop_last"] = float(new_stop)

                    if abs(broker.pos_qty) < 1e-12:
                        finalize_trade(exit_dt=dt, reason="tp_final")
                        baseline_qty_abs = 0.0
                        tps_hit = 0

                        guard2 = active_plan.guardrails if (active_plan and active_plan.guardrails) else None
                        if trades and trades[-1]["exit_dt"] is not None:
                            apply_cooldowns_after_trade(dt_ns, float(trades[-1]["net_pnl"]), guard2)
                        break

        guard = active_plan.guardrails if (active_plan and active_plan.guardrails) else None
        eq_close = broker.equity(mc)

        if guard is not None:
            if guard.min_equity_stop is not None and eq_close < float(guard.min_equity_stop):
                if abs(broker.pos_qty) >= 1e-12:
                    broker.trade_to_target_qty(
                        dt=dt,
                        target_qty=0.0,
                        ref_price=float(c),
                        order_type=OrderType.MARKET,
                        slip_bps=slip_bps_for(cfg, OrderType.MARKET, vol_bps, liq_mult),
                        spread_bps=spread_bps_for(cfg, vol_bps, liq_mult),
                        liq_mult=liq_mult,
                        force_close=True,
                    )
                    finalize_trade(exit_dt=dt, reason="kill")

                broker.cashflow_this_bar = float(cashflow_this_bar)
                broker.mark(dt, mc)
                break

            if guard.max_daily_loss_pct is not None and day_start_equity is not None:
                if eq_close <= float(day_start_equity) * (1.0 - float(guard.max_daily_loss_pct)):
                    if daily_blocked_date != current_day:
                        daily_blocked_date = current_day
                        blocked_days.add(current_day)
                        daily_loss_triggers += 1

                    if guard.close_on_daily_limit and abs(broker.pos_qty) >= 1e-12:
                        broker.trade_to_target_qty(
                            dt=dt,
                            target_qty=0.0,
                            ref_price=float(c),
                            order_type=OrderType.MARKET,
                            slip_bps=slip_bps_for(cfg, OrderType.MARKET, vol_bps, liq_mult),
                            spread_bps=spread_bps_for(cfg, vol_bps, liq_mult),
                            liq_mult=liq_mult,
                            force_close=True,
                        )
                        finalize_trade(exit_dt=dt, reason="daily_loss_limit")

        if i == len(df) - 1 and cfg.close_open_at_end and abs(broker.pos_qty) >= 1e-12:
            broker.trade_to_target_qty(
                dt=dt,
                target_qty=0.0,
                ref_price=float(c),
                order_type=OrderType.MARKET,
                slip_bps=slip_bps_for(cfg, OrderType.MARKET, vol_bps, liq_mult),
                spread_bps=spread_bps_for(cfg, vol_bps, liq_mult),
                liq_mult=liq_mult,
                force_close=True,
            )
            finalize_trade(exit_dt=dt, reason="eod")
        
        broker.cashflow_this_bar = float(cashflow_this_bar)
        broker.mark(dt, mc)

        # --- PATCH: FeatureView (no per-bar dict allocation) ---
        feature_view.set_index(i)

        ctx = StrategyContext(
            candle=Candle(
                dt=dt,
                ts=int(ts[i]) if ts is not None else None,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=v,
                vol_bps=vol_bps,
                liq_mult=liq_mult,
            ),
            account=account_state(mc),
            position=position_state(mc),
            constraints=constraints,
            features=feature_view,  # supports ctx.features.get(...)
            last_exec=last_exec,
        )
        pending_update = strategy.on_bar(ctx)
        if pending_update is None:
            pending_update = PlanUpdate(action=PlanAction.HOLD)

    if show_progress and (not use_tqdm):
        sys.stdout.write("\rBacktest progress: 100%                       \n")
        sys.stdout.flush()

    fills_df = pd.DataFrame(broker.fills)
    eq_df = pd.DataFrame(broker.equity_curve)
    trades_df = pd.DataFrame(trades)

    fj = pd.Series(fund_jitter_fracs, dtype=float)

    exec_stats = {
        "seed": int(seed),
        "liquidated": bool(liquidated),
        "tp_limit_attempts": int(tp_limit_attempts),
        "tp_limit_fills": int(tp_limit_fills),
        "tp_limit_misses": int(tp_limit_misses),
        "fund_jitter_events": int(len(fund_jitter_fracs)),
        "funding_events_real": int(funding_events_real),
        "fund_jitter_avg_frac": float(fj.mean()) if len(fj) else 0.0,
        "fund_jitter_p90_frac": float(fj.quantile(0.9)) if len(fj) else 0.0,
        "fund_jitter_max_frac": float(fj.max()) if len(fj) else 0.0,
        "price_tick_rounds": int(broker.price_tick_rounds),
        "rej_qty_step": int(broker.rejected_qty_step),
        "rej_min_notional": int(broker.rejected_min_notional),
    }

    guardrail_stats = {
        "daily_loss_triggers": int(daily_loss_triggers),
        "blocked_days": int(len(blocked_days)),
        "cooldown_triggers": int(cooldown_triggers),
        "max_loss_streak": int(max_loss_streak),
    }

    final_eq = (
        float(eq_df["equity"].iloc[-1]) if len(eq_df) else broker.equity(float(mark_closes[-1]))
    )
    end_over_start_return = (
        final_eq / float(starting_equity) - 1.0 if starting_equity > 0 else 0.0
    )

    cashflow_total = 0.0
    if eq_df is not None and not eq_df.empty and "cashflow" in eq_df.columns:
        cashflow_total = float(
            pd.to_numeric(eq_df["cashflow"], errors="coerce").fillna(0.0).sum()
        )

    # Spot/DCA: total_return should be TWR (deposit-neutral) if available.
    # Perps: keep end/start return semantics.
    total_return = float(end_over_start_return)
    if is_spot:
        try:
            perf_tmp = _build_cashflow_performance_stats(df, eq_df)
            if isinstance(perf_tmp, dict) and "twr_total_return" in perf_tmp:
                total_return = float(perf_tmp.get("twr_total_return", 0.0) or 0.0)
        except Exception:
            pass

    net_profit_ex_cashflows = float(final_eq - float(starting_equity) - cashflow_total)

    metrics = {
        "equity": {
            "start": float(starting_equity),
            "end": float(final_eq),
            "total_return": float(total_return),
            "end_over_start_return": float(end_over_start_return),
            "cashflow_total": float(cashflow_total),
            "net_profit_ex_cashflows": float(net_profit_ex_cashflows),
        },
        "guardrails": guardrail_stats,
        "execution": exec_stats,
    }

    return metrics, fills_df, eq_df, trades_df, guardrail_stats


# ============================================================
# CLI / runner
# ============================================================


def _load_ohlcv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")

    suf = p.suffix.lower()
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(p)
    if suf == ".csv":
        return pd.read_csv(p)

    raise ValueError(f"Unsupported file type: {p.suffix} (use .csv or .parquet)")

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip() for c in df.columns})

    alias = {
        "timestamp": "ts",
        "time": "dt",
        "datetime": "dt",
        "date": "dt",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }
    for k, v in alias.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    required_any = {"open", "high", "low", "close"}
    missing = required_any - set(df.columns)
    if missing:
        raise ValueError(f"Data missing required OHLC columns: {sorted(missing)}")

    if "dt" not in df.columns:
        if "ts" not in df.columns:
            raise ValueError("Data must include 'dt' (or 'ts' to derive dt).")

        ts = pd.to_numeric(df["ts"], errors="coerce")
        ts_clean = ts.dropna().astype(np.int64)
        if len(ts_clean) == 0:
            raise ValueError("ts column exists but could not be parsed as numbers.")

        unit = "ms" if int(ts_clean.median()) > 10_000_000_000 else "s"
        df["dt"] = pd.to_datetime(ts, unit=unit, utc=True).astype(str)

    dt_parsed = pd.to_datetime(df["dt"], utc=True, errors="coerce")
    if dt_parsed.isna().any():
        bad = df.loc[dt_parsed.isna(), "dt"].head(5).tolist()
        raise ValueError(f"Failed to parse some dt values (first 5): {bad}")
    df["dt"] = dt_parsed.astype(str)

    if "ts" not in df.columns:
        df["ts"] = (dt_parsed.astype("int64") // 1_000_000).astype(np.int64)

    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "dt", "ts"]).copy()
    df = df.sort_values("ts").reset_index(drop=True)

    return df


def _ensure_vol_bps(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    if "vol_bps" in df.columns:
        df["vol_bps"] = pd.to_numeric(df["vol_bps"], errors="coerce").fillna(0.0)
        return df

    w = int(max(2, window))
    c = df["close"].astype(float)
    r = np.log(c).diff()
    vol = r.rolling(w, min_periods=max(2, w // 2)).std()
    df["vol_bps"] = (vol.fillna(0.0) * 10_000.0).astype(float)
    return df


def _import_strategy(dotted: str):
    if ":" in dotted:
        mod_name, obj_name = dotted.split(":", 1)
    else:
        mod_name, obj_name = dotted, "Strategy"

    mod = importlib.import_module(mod_name)
    if not hasattr(mod, obj_name):
        raise AttributeError(f"Strategy '{obj_name}' not found in module '{mod_name}'")
    cls_or_obj = getattr(mod, obj_name)
    return cls_or_obj() if callable(cls_or_obj) else cls_or_obj


def _make_constraints() -> EngineConstraints:
    return EngineConstraints(
        max_leverage=20.0,
        maint_margin_rate=0.005,
        price_tick=0.1,
        qty_step=0.001,
        min_notional_usdt=5.0,
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Plan-driven backtester (self-managed)")
    ap.add_argument(
        "--data",
        required=True,
        help="Path to OHLCV CSV/Parquet (must include dt or ts + OHLC)",
    )
    ap.add_argument(
        "--strategy",
        required=True,
        help="Dotted path: module:Class  (e.g. strategies.ta_v1:TAV1Strategy)",
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--starting-equity", type=float, default=1000.0)
    ap.add_argument(
        "--vol-window",
        type=int,
        default=60,
        help="Rolling window (bars) for vol_bps if missing",
    )
    ap.add_argument("--out", default="runs", help="Output folder (default: runs)")
    ap.add_argument(
        "--run-name",
        default=None,
        help="Run name subfolder (default: timestamp_strategy)",
    )
    ap.add_argument(
        "--no-progress", action="store_true", help="Disable progress display"
    )
    args = ap.parse_args(argv)

    df = _load_ohlcv(args.data)
    df = _normalize_columns(df)
    df = _ensure_vol_bps(df, window=args.vol_window)

    if "liq_mult" not in df.columns:
        df["liq_mult"] = 1.0
    else:
        df["liq_mult"] = pd.to_numeric(df["liq_mult"], errors="coerce").fillna(1.0)

    strategy = _import_strategy(args.strategy)
    constraints = _make_constraints()
    cfg = BacktestConfig(market_mode="spot")

    strat_tag = args.strategy.replace(":", "_").replace(".", "_")
    run_name = args.run_name or f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{strat_tag}"
    out_dir = (Path(args.out) / run_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    metrics, fills_df, eq_df, trades_df, _guard = run_backtest_once(
        df=df,
        strategy=strategy,
        seed=int(args.seed),
        starting_equity=float(args.starting_equity),
        constraints=constraints,
        cfg=cfg,
        verbose=False,
        show_progress=(not args.no_progress),
        features_ready=False,
    )
    elapsed = time.time() - t0

    # --- rich reporting layer ---
    bars = int(len(eq_df)) if eq_df is not None else 0
    dt_start = (
        str(eq_df["dt"].iloc[0])
        if (eq_df is not None and bars and "dt" in eq_df.columns)
        else "n/a"
    )
    dt_end = (
        str(eq_df["dt"].iloc[-1])
        if (eq_df is not None and bars and "dt" in eq_df.columns)
        else "n/a"
    )

    if str(getattr(cfg, "market_mode", "perps")).lower() == "spot":
        perf = _build_cashflow_performance_stats(df, eq_df)
    else:
        perf = _build_performance_stats(df, eq_df)
    tstats = _trade_stats(trades_df)
    exits = _exit_counts(trades_df)
    tp_levels = _tp_level_counts(trades_df)
    stop_moves = _stop_move_stats(trades_df)
    efficiency = _efficiency_stats(trades_df)

    exec_break = _execution_breakdown(fills_df)
    exposure = _exposure_stats(eq_df, fills_df)

    fees_total = (
        float(eq_df["fees_paid_total"].iloc[-1])
        if (eq_df is not None and bars and "fees_paid_total" in eq_df.columns)
        else 0.0
    )
    funding_net = (
        float(eq_df["funding_net"].iloc[-1])
        if (eq_df is not None and bars and "funding_net" in eq_df.columns)
        else 0.0
    )

    metrics.setdefault("runtime", {})
    metrics["runtime"].update(
        {
            "bars": bars,
            "dt_start": dt_start,
            "dt_end": dt_end,
            "elapsed_sec": float(elapsed),
        }
    )
    metrics["performance"] = perf
    metrics["trades_summary"] = tstats
    metrics["exit_counts"] = exits
    metrics["tp_level_counts"] = tp_levels
    metrics["stop_move_stats"] = stop_moves
    metrics["efficiency"] = efficiency
    metrics["execution_breakdown"] = exec_break
    metrics["exposure"] = exposure
    metrics.setdefault("costs", {})
    metrics["costs"].update(
        {"fees_paid_total": float(fees_total), "funding_net": float(funding_net)}
    )

    metrics["tp_sl_profitability"] = _tp_sl_profitability_report(trades_df)

    try:
        if (
            eq_df is not None
            and not eq_df.empty
            and trades_df is not None
            and not trades_df.empty
            and "equity" in eq_df.columns
        ):
            eq_delta = float(eq_df["equity"].iloc[-1]) - float(eq_df["equity"].iloc[0])
            trade_sum = (
                float(
                    pd.to_numeric(trades_df["net_pnl"], errors="coerce")
                    .fillna(0.0)
                    .sum()
                )
                if "net_pnl" in trades_df.columns
                else 0.0
            )
            cashflow_total = 0.0
            if "cashflow" in eq_df.columns:
                cashflow_total = float(
                    pd.to_numeric(eq_df["cashflow"], errors="coerce").fillna(0.0).sum()
                )

            is_spot = str(getattr(cfg, "market_mode", "perps")).lower() == "spot"
            diff = float(eq_delta - trade_sum)
            if is_spot:
                diff = float(eq_delta - cashflow_total - trade_sum)

            metrics["reconciliation"] = {
                "equity_delta": float(eq_delta),
                "sum_trade_net_pnl": float(trade_sum),
                "cashflow_total": float(cashflow_total),
                "diff": float(diff),
            }
    except Exception:
        metrics["reconciliation"] = {"error": "failed_to_compute"}

    _write_json(out_dir / "metrics.json", metrics)
    if fills_df is not None and not fills_df.empty:
        fills_df.to_csv(out_dir / "fills.csv", index=False)
    if eq_df is not None and not eq_df.empty:
        eq_df.to_csv(out_dir / "equity_curve.csv", index=False)
    if trades_df is not None and not trades_df.empty:
        trades_df.to_csv(out_dir / "trades.csv", index=False)

    _save_daily_monthly_returns(out_dir, eq_df)
    _write_json(out_dir / "report.json", metrics)

    # --- console summary ---
    eq = metrics["equity"]
    ret = float(eq["total_return"])
    mdd = float(perf.get("max_drawdown", 0.0))

    ann_ret = float(perf.get("annualized_return", 0.0))
    ann_vol = float(perf.get("annualized_vol", 0.0))
    sharpe = float(perf.get("sharpe", 0.0))
    sortino = float(perf.get("sortino", 0.0))
    calmar = perf.get("calmar", 0.0)

    pf = tstats.get("profit_factor", 0.0)
    pf_str = (
        "inf"
        if (isinstance(pf, float) and (not math.isfinite(pf)) and pf > 0)
        else f"{pf:.3f}"
    )

    top_exits = sorted(exits.items(), key=lambda kv: kv[1], reverse=True)[:8]
    exits_line = ", ".join([f"{k}:{v}" for k, v in top_exits]) if top_exits else "n/a"

    # Truth detector fields
    fee_impact = float(efficiency.get("fee_impact_pct", 0.0) or 0.0)
    sqn = float(tstats.get("sqn", 0.0) or 0.0)
    avg_r = float(tstats.get("avg_r", 0.0) or 0.0)
    avg_dur_win = float(tstats.get("avg_duration_win_min", 0.0) or 0.0)
    avg_dur_loss = float(tstats.get("avg_duration_loss_min", 0.0) or 0.0)
    dur_ratio = (avg_dur_win / avg_dur_loss) if avg_dur_loss > 0 else 0.0

    summary_lines: List[str] = []
    summary_lines.append("=== Backtest Complete ===")
    summary_lines.append(f"Data:      {args.data}")
    summary_lines.append(f"Strategy:  {args.strategy}")
    summary_lines.append(f"Seed:      {args.seed}")
    summary_lines.append(f"Bars:      {bars}  ({dt_start} -> {dt_end})")
    summary_lines.append(f"Elapsed:   {elapsed:.2f}s")
    summary_lines.append(
        f"Equity:    {eq['start']:.2f} -> {eq['end']:.2f}  "
        f"(return {_format_pct(ret)})"
    )
    # Spot/DCA: show deposits and net profit excluding deposits for interpretability.
    if str(getattr(cfg, "market_mode", "perps")).lower() == "spot":
        cashflow_total = float(eq.get("cashflow_total", 0.0) or 0.0)
        eos_ret = float(eq.get("end_over_start_return", 0.0) or 0.0)
        net_ex = float(eq.get("net_profit_ex_cashflows", 0.0) or 0.0)
        summary_lines.append(
            f"Deposits:  {_format_money(cashflow_total)}   "
            f"End/Start: {_format_pct(eos_ret)}   "
            f"Net(ex dep): {net_ex:+.2f}"
        )
    summary_lines.append(f"Max DD:    {_format_pct(mdd)}")
    summary_lines.append(
        f"Ann Ret:   {_format_pct(ann_ret)}   Ann Vol: {_format_pct(ann_vol)}"
    )
    summary_lines.append(
        f"Sharpe:    {sharpe:.3f}   Sortino: {sortino:.3f}   Calmar: {calmar}"
    )
    summary_lines.append(
        f"Time in Mkt: {_format_pct(float(exposure.get('time_in_market_frac', 0.0)))}   "
        f"Turnover: {float(exposure.get('turnover_notional_over_avg_equity', 0.0)):.2f}x"
    )
    summary_lines.append("-" * 30)
    summary_lines.append(">>> TRUTH DETECTORS <<<")
    summary_lines.append("SQN:       {:.2f}  (>2.0 is good, >5.0 is elite)".format(sqn))
    summary_lines.append(f"Avg R:     {avg_r:.2f}R")
    summary_lines.append(
        f"Win/Loss Dur: {avg_dur_win:.1f}m / {avg_dur_loss:.1f}m  "
        f"(Ratio: {dur_ratio:.2f}x)"
    )
    summary_lines.append(f"Fee Impact: {fee_impact:.1f}% of gross PnL")
    summary_lines.append("-" * 30)
    summary_lines.append(f"Trades:    {tstats['trades_closed']}")
    summary_lines.append(
        f"Wins:      {tstats['wins']} / {tstats['trades_closed']}  "
        f"(win rate {_format_pct(tstats['win_rate'])})"
    )
    summary_lines.append(
        f"Avg PnL:   {tstats['avg_pnl']:+.2f}   Median: {tstats['median_pnl']:+.2f}"
    )
    summary_lines.append(
        f"Avg Win:   {tstats['avg_win']:+.2f}   Avg Loss: {tstats['avg_loss']:+.2f}"
    )
    summary_lines.append(f"PF:        {pf_str}")
    summary_lines.append(
        f"Best/Worst:{tstats['best_trade']:+.2f} / {tstats['worst_trade']:+.2f}"
    )
    summary_lines.append(
        f"Streaks:   W {tstats['max_consec_wins']}   L {tstats['max_consec_losses']}"
    )
    summary_lines.append(f"Fees:      {_format_money(fees_total)}   Funding: {funding_net:+.2f}")

    if exec_break:
        summary_lines.append(
            "Exec:      "
            f"fills={exec_break.get('fills',0)}  "
            f"notional={_format_money(exec_break.get('total_notional',0.0))}  "
            f"avg_slip={exec_break.get('weighted_avg_slip_bps',0.0):.2f}bps  "
            f"avg_spread={exec_break.get('weighted_avg_spread_bps',0.0):.2f}bps"
        )
        summary_lines.append(
            "Maker/Taker: "
            f"maker_fees={_format_money(exec_break['maker']['fees'])}  "
            f"taker_fees={_format_money(exec_break['taker']['fees'])}"
        )

    if tp_levels:
        keys = [k for k in tp_levels.keys() if k.startswith("tp") and k.endswith("_hit_trades")]

        def _tp_key(s: str) -> int:
            try:
                mid = s[2 : s.index("_")]
                return int(mid)
            except Exception:
                return 999999

        keys = sorted(keys, key=_tp_key)
        tp_parts = [f"TP{_tp_key(k)}:{tp_levels[k]}" for k in keys if _tp_key(k) != 999999]
        if tp_parts:
            summary_lines.append("TP Hits:   " + "  ".join(tp_parts))

    if stop_moves:
        summary_lines.append(
            "Stops:     "
            f"moves={stop_moves.get('stop_move_events',0)} "
            f"(tp={stop_moves.get('stop_move_tp_events',0)}, "
            f"plan={stop_moves.get('stop_move_plan_events',0)})  "
            f"stop_hits(moved/orig)={stop_moves.get('stop_hits_after_move',0)}/"
            f"{stop_moves.get('stop_hits_original',0)}"
        )

    if (
        "reconciliation" in metrics
        and isinstance(metrics["reconciliation"], dict)
        and "diff" in metrics["reconciliation"]
    ):
        summary_lines.append(
            f"Recon:     diff={float(metrics['reconciliation']['diff']):+.6f}"
        )

    summary_lines.append(f"Exits:     {exits_line}")
    summary_lines.append(f"Outputs:   {out_dir}")

    summary_text = "\n".join(summary_lines) + "\n"
    print("\n" + summary_text)
    _write_text(out_dir / "summary.txt", summary_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())