from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from lab.metrics import METRICS, get_metric_value
from lab.profile_default import Constraint, Question


@dataclass(frozen=True)
class Alert:
    severity: str  # "info" | "warn" | "critical"
    question_id: str
    metric_id: str
    metric_value: float
    op: str
    threshold: float
    message: str
    used_fallback: bool = False


def _passes(op: str, value: float, threshold: float) -> bool:
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    return True


def evaluate_constraints(
    row: Dict[str, Any],
    *,
    question_id: str,
    constraints: List[Constraint],
) -> Tuple[List[Alert], bool, bool]:
    alerts: List[Alert] = []
    used_fallback_any = False
    missing_any = False

    for c in constraints:
        v = get_metric_value(row, c.metric_id)
        used_fallback = False
        metric_id_used = c.metric_id

        if v != v and c.fallback_metric_id:
            v2 = get_metric_value(row, c.fallback_metric_id)
            if v2 == v2:
                v = v2
                used_fallback = True
                used_fallback_any = True
                metric_id_used = c.fallback_metric_id

        # If still missing, skip (or could produce "needs rolling-starts" prompt later)
        if v != v:
            missing_any = True
            continue

        ok = _passes(c.op, v, float(c.threshold))
        if ok:
            continue

        spec = METRICS.get(metric_id_used)
        label = spec.label if spec else metric_id_used
        fmt = spec.fmt if spec else (lambda x: str(x))
        msg = (
            f"{label} is {fmt(float(v))} but your limit is {c.op} {fmt(float(c.threshold))}."
        )

        alerts.append(
            Alert(
                severity=str(c.severity),
                question_id=str(question_id),
                metric_id=str(metric_id_used),
                metric_value=float(v),
                op=str(c.op),
                threshold=float(c.threshold),
                message=msg,
                used_fallback=bool(used_fallback),
            )
        )

    return alerts, used_fallback_any, missing_any


def evaluate_profile(
    row: Dict[str, Any],
    questions: List[Question],
    answers: Dict[str, int],
) -> Dict[str, Any]:
    violations: List[Alert] = []
    evidence: List[Alert] = []

    for q in questions:
        idx = int(answers.get(q.id, q.default_index))
        idx = max(0, min(idx, len(q.choices) - 1))
        choice = q.choices[idx]
        viols, used_fb, missing = evaluate_constraints(
            row, question_id=q.id, constraints=choice.constraints
        )
        violations.extend(viols)

        # Evidence flags: nudge toward rolling-start measurement / clarify fallbacks.
        if choice.constraints:
            if used_fb:
                evidence.append(
                    Alert(
                        severity="info",
                        question_id=str(q.id),
                        metric_id="(fallback)",
                        metric_value=float("nan"),
                        op="",
                        threshold=float("nan"),
                        message="Used a single-run estimate for this preference (rolling-start not available).",
                        used_fallback=True,
                    )
                )
            if missing and (not used_fb):
                evidence.append(
                    Alert(
                        severity="info",
                        question_id=str(q.id),
                        metric_id="(missing)",
                        metric_value=float("nan"),
                        op="",
                        threshold=float("nan"),
                        message="Not measured for this preference (missing rolling-start data).",
                        used_fallback=False,
                    )
                )

    sev_rank = {"critical": 2, "warn": 1}
    max_violation_sev = ""
    if violations:
        max_violation_sev = max(
            violations, key=lambda a: sev_rank.get(a.severity, 0)
        ).severity

    top_reason = ""
    if violations:
        crits = [a for a in violations if a.severity == "critical"]
        a0 = crits[0] if crits else violations[0]
        spec = METRICS.get(a0.metric_id)
        label = spec.label if spec else a0.metric_id
        top_reason = f"{label} failed your tolerance"
    elif evidence:
        top_reason = "Needs more evidence for your preferences"

    return {
        "violations.count": int(len(violations)),
        "violations.max_severity": str(max_violation_sev),
        "evidence.count": int(len(evidence)),
        "alerts.top_reason": str(top_reason),
        "violations.list": violations,
        "evidence.list": evidence,
    }