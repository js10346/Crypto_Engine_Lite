from __future__ import annotations

from engine.contracts import OrderType, PlanAction, PlanUpdate, TradePlan


class Strategy:
    def __init__(self):
        self.did_buy = False

    def on_bar(self, ctx):
        if self.did_buy:
            return PlanUpdate(action=PlanAction.HOLD)

        price = float(ctx.candle.open)
        if price <= 0:
            return PlanUpdate(action=PlanAction.HOLD)

        usd = 200.0
        qty = usd / price

        self.did_buy = True
        return PlanUpdate(
            action=PlanAction.REPLACE,
            plan=TradePlan(
                desired_side=1,
                target_qty=float(qty),
                entry_order_type=OrderType.MARKET,
                metadata={"note": "smoke_spot_buy_once"},
            ),
        )