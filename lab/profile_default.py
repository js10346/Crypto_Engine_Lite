from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Constraint:
    metric_id: str
    op: str  # ">=", "<="
    threshold: float
    severity: str  # "warn" | "critical"
    fallback_metric_id: Optional[str] = None
    note: str = ""


@dataclass(frozen=True)
class Choice:
    label: str
    constraints: List[Constraint]


@dataclass(frozen=True)
class Question:
    id: str
    title: str
    explanation: str
    choices: List[Choice]
    default_index: int = 0


def default_questions() -> List[Question]:
    return [
        Question(
            id="start_bad_time_loss",
            title="If you started at a bad time, how much loss could you tolerate?",
            explanation=(
                "We simulate starting the same plan on many different start dates. "
                "This checks the worst 10% outcome (p10) of time‑weighted return."
            ),
            choices=[
                Choice(
                    "I don't want the worst 10% to lose money",
                    [Constraint("twr_p10", ">=", 0.0, "critical", fallback_metric_id="performance.twr_total_return")],
                ),
                Choice(
                    "I can tolerate up to -10%",
                    [Constraint("twr_p10", ">=", -0.10, "warn", fallback_metric_id="performance.twr_total_return")],
                ),
                Choice(
                    "I can tolerate up to -25%",
                    [Constraint("twr_p10", ">=", -0.25, "warn", fallback_metric_id="performance.twr_total_return")],
                ),
                Choice(
                    "I can tolerate up to -50%",
                    [Constraint("twr_p10", ">=", -0.50, "warn", fallback_metric_id="performance.twr_total_return")],
                ),
                Choice("Don't filter on this", []),
            ],
            default_index=1,
        ),
        Question(
            id="drawdown_tolerance",
            title="How big of a drop from a previous high could you tolerate?",
            explanation=(
                "This uses the p90 drawdown across rolling starts (bad-but-common). "
                "Lower is easier to live with."
            ),
            choices=[
                Choice("Max 20% drop", [Constraint("dd_p90", "<=", 0.20, "critical", fallback_metric_id="performance.max_drawdown_equity")]),
                Choice("Max 35% drop", [Constraint("dd_p90", "<=", 0.35, "warn", fallback_metric_id="performance.max_drawdown_equity")]),
                Choice("Max 50% drop", [Constraint("dd_p90", "<=", 0.50, "warn", fallback_metric_id="performance.max_drawdown_equity")]),
                Choice("Max 70% drop", [Constraint("dd_p90", "<=", 0.70, "warn", fallback_metric_id="performance.max_drawdown_equity")]),
                Choice("Don't filter on this", []),
            ],
            default_index=1,
        ),
        Question(
            id="typical_positive",
            title="Do you need the typical outcome to be positive?",
            explanation="This checks the median (p50) time‑weighted return across rolling starts.",
            choices=[
                Choice("Yes (median must be positive)", [Constraint("twr_p50", ">=", 0.0, "critical", fallback_metric_id="performance.twr_total_return")]),
                Choice("No (I'm okay if it's mixed)", []),
            ],
            default_index=0,
        ),
        Question(
            id="fees_churn",
            title="How sensitive are you to fees and frequent trading?",
            explanation=(
                "This checks turnover and fee impact. More churn usually means "
                "more slippage/fees and more fragility in real life."
            ),
            choices=[
                Choice(
                    "Very fee-sensitive",
                    [
                        Constraint("exposure.turnover_notional_over_avg_equity", "<=", 0.5, "warn"),
                        Constraint("efficiency.fee_impact_pct", "<=", 10.0, "warn"),
                    ],
                ),
                Choice(
                    "Moderately fee-sensitive",
                    [
                        Constraint("exposure.turnover_notional_over_avg_equity", "<=", 1.5, "warn"),
                        Constraint("efficiency.fee_impact_pct", "<=", 25.0, "warn"),
                    ],
                ),
                Choice("Not very fee-sensitive", [Constraint("exposure.turnover_notional_over_avg_equity", "<=", 3.0, "warn")]),
                Choice("Don't filter on this", []),
            ],
            default_index=1,
        ),

        Question(
            id="underwater_tolerance",
            title="How long can you tolerate being underwater (below a previous high)?",
            explanation=(
                "This checks the p90 underwater duration across rolling starts. "
                "Underwater time is how long the equity stays below its previous peak."
            ),
            choices=[
                Choice("About 1 month", [Constraint("uw_p90_days", "<=", 30.0, "critical")]),
                Choice("About 3 months", [Constraint("uw_p90_days", "<=", 90.0, "warn")]),
                Choice("About 6 months", [Constraint("uw_p90_days", "<=", 180.0, "warn")]),
                Choice("About 1 year", [Constraint("uw_p90_days", "<=", 365.0, "warn")]),
                Choice("1–2 years is fine", [Constraint("uw_p90_days", "<=", 730.0, "info")]),
                Choice("Don't filter on this", []),
            ],
            default_index=3,
        ),

        Question(
            id="cash_deployment",
            title="Do you want this plan invested most of the time, or mostly in stablecoins?",
            explanation=(
                "This checks the median invested fraction over time. "
                "If it's low, the plan often sits in cash and may not behave like a true DCA."
            ),
            choices=[
                Choice("Mostly invested", [Constraint("util_p50", ">=", 0.75, "warn")]),
                Choice("Balanced (about half invested)", [Constraint("util_p50", ">=", 0.50, "info")]),
                Choice("Mostly in cash is fine", [Constraint("util_p50", ">=", 0.25, "info")]),
                Choice("I don't care", []),
            ],
            default_index=1,
        ),
    ]