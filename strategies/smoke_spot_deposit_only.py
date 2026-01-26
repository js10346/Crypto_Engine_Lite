from __future__ import annotations

from engine.contracts import PlanAction, PlanUpdate, TradePlan


class Strategy:
    def __init__(self):
        self.i = 0

    def on_bar(self, ctx):
        # Deposit $10 every day for first 30 bars, never buy
        cd = 10.0 if self.i < 30 else 0.0
        self.i += 1

        return PlanUpdate(
            action=PlanAction.REPLACE if cd != 0 else PlanAction.HOLD,
            plan=TradePlan(desired_side=1, target_qty=0.0, cash_delta=cd)
            if cd != 0
            else None,
        )