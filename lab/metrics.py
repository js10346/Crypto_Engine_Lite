from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


Formatter = Callable[[float], str]


def fmt_pct(x: float) -> str:
    return "n/a" if (not math.isfinite(x)) else f"{x * 100:.2f}%"


def fmt_money(x: float) -> str:
    return "n/a" if (not math.isfinite(x)) else f"{x:,.2f}"


def fmt_days(x: float) -> str:
    return "n/a" if (not math.isfinite(x)) else f"{x:.0f} days"


def fmt_num(x: float) -> str:
    return "n/a" if (not math.isfinite(x)) else f"{x:.4f}"


@dataclass(frozen=True)
class MetricSpec:
    id: str
    label: str
    unit: str
    direction: str  # "higher_is_better" | "lower_is_better"
    fmt: Formatter
    help: str = ""


METRICS: Dict[str, MetricSpec] = {
    # Rolling-start summary (distribution stats)
    "twr_p10": MetricSpec(
        id="twr_p10",
        label="Worst-decile time-weighted return (p10)",
        unit="fraction",
        direction="higher_is_better",
        fmt=fmt_pct,
        help=(
            "Rolling-start p10 time-weighted return. "
            "If this is negative, the plan often loses money depending on start date."
        ),
    ),
    "twr_p50": MetricSpec(
        id="twr_p50",
        label="Median time-weighted return (p50)",
        unit="fraction",
        direction="higher_is_better",
        fmt=fmt_pct,
        help="Rolling-start median time-weighted return.",
    ),
    "dd_p90": MetricSpec(
        id="dd_p90",
        label="High drawdown scenario (p90)",
        unit="fraction",
        direction="lower_is_better",
        fmt=fmt_pct,
        help=(
            "Rolling-start p90 max drawdown. "
            "This approximates 'bad-but-common' drawdown severity."
        ),
    ),

    "uw_p90_days": MetricSpec(
        id="uw_p90_days",
        label="Underwater time (p90)",
        unit="days",
        direction="lower_is_better",
        fmt=fmt_days,
        help="Rolling-start p90 of longest underwater streak (days below prior high).",
    ),
    "util_p50": MetricSpec(
        id="util_p50",
        label="Typical invested fraction (p50)",
        unit="fraction",
        direction="higher_is_better",
        fmt=fmt_pct,
        help="Rolling-start median mean utilization: average invested fraction over time.",
    ),

    "profit_p10": MetricSpec(
        id="profit_p10",
        label="Worst-decile profit (excluding deposits)",
        unit="usd",
        direction="higher_is_better",
        fmt=fmt_money,
        help=(
            "Rolling-start p10 profit excluding deposits. "
            "Useful only when comparing similar deposit schedules."
        ),
    ),
    "robustness_score": MetricSpec(
        id="robustness_score",
        label="Robustness score",
        unit="score",
        direction="higher_is_better",
        fmt=fmt_num,
        help="A simple composite from rolling-start stats (you can change it later).",
    ),
    # Full results metrics (single-run, deposit-aware)
    "equity.net_profit_ex_cashflows": MetricSpec(
        id="equity.net_profit_ex_cashflows",
        label="Net profit (excluding deposits)",
        unit="usd",
        direction="higher_is_better",
        fmt=fmt_money,
        help="Final equity minus starting equity minus deposits.",
    ),
    "performance.max_drawdown_equity": MetricSpec(
        id="performance.max_drawdown_equity",
        label="Max drawdown (single run)",
        unit="fraction",
        direction="lower_is_better",
        fmt=fmt_pct,
        help="Max drawdown on equity curve for the single run.",
    ),
    "performance.twr_total_return": MetricSpec(
        id="performance.twr_total_return",
        label="Time-weighted return (single run)",
        unit="fraction",
        direction="higher_is_better",
        fmt=fmt_pct,
        help="Deposit-neutral return for the single run.",
    ),
    "exposure.time_in_market_frac": MetricSpec(
        id="exposure.time_in_market_frac",
        label="Time in market",
        unit="fraction",
        direction="higher_is_better",
        fmt=fmt_pct,
        help="Fraction of bars with any position.",
    ),
    "exposure.turnover_notional_over_avg_equity": MetricSpec(
        id="exposure.turnover_notional_over_avg_equity",
        label="Turnover (period)",
        unit="x",
        direction="lower_is_better",
        fmt=fmt_num,
        help="Total traded notional divided by average equity over the tested period.",
    ),
    "efficiency.fee_impact_pct": MetricSpec(
        id="efficiency.fee_impact_pct",
        label="Fee impact (vs gross PnL)",
        unit="percent",
        direction="lower_is_better",
        fmt=lambda x: "n/a" if (not math.isfinite(x)) else f"{x:.1f}%",
        help="Fees as a percentage of absolute gross PnL in the single run.",
    ),
}


def get_metric_value(row: Dict[str, Any], metric_id: str) -> float:
    v = row.get(metric_id, float("nan"))
    try:
        x = float(v)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")