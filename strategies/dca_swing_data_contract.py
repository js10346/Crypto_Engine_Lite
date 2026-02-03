from __future__ import annotations

import math
from typing import Optional, Tuple

from engine.contracts import (
    OrderType,
    PlanAction,
    PlanUpdate,
    StrategyConfig,
    StrategyContext,
    TradePlan,
)


def _freq_to_days(freq: str) -> int:
    f = str(freq or "none").strip().lower()
    if f in {"none", "off", "0", ""}:
        return 0
    if f in {"daily", "day", "1d"}:
        return 1
    if f in {"weekly", "week", "7d"}:
        return 7
    if f in {"monthly", "month", "30d"}:
        return 30
    # Default: treat unknown as "none"
    return 0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


DEFAULT_CONFIG = StrategyConfig(
    strategy_name="dca_swing",
    side="long",
    params={
        # Deposit (external cashflows into stablecoin cash)
        "deposit_freq": "weekly",  # none|daily|weekly|monthly
        "deposit_amount_usd": 50.0,
        # Buy cadence (deploy from cash pile)
        "buy_freq": "weekly",  # daily|weekly|monthly
        "buy_amount_usd": 50.0,
        # Buy filter
        "buy_filter": "none",  # none|below_ema|rsi_below
        "ema_len": 200,
        "rsi_thr": 40.0,
        # Allocation cap
        "max_alloc_pct": 1.0,
        # Risk controls
        "sl_pct": 0.0,  # fraction (0.10 = 10%); stop at avg_entry*(1-sl_pct); 0 disables
        "tp_pct": 0.0,  # fraction (0.10 = 10%); take profit at avg_entry*(1+tp_pct); 0 disables
        "tp_sell_fraction": 0.50,  # fraction of position to sell on TP hit
        "reserve_frac_of_proceeds": 0.50,  # portion of TP proceeds to reserve in cash
    },
)


class DCASwingStrategy:
    """
    Daily-bar DCA + Swing overlay template.

    Key behaviors:
    - External deposits always go to cash first via TradePlan.cash_delta (applied at next bar open).
    - Buys use "available cash" = (cash - reserved_cash) plus any deposit scheduled for next open.
    - Sells are long-only reductions of position (never short).
    - SL is an engine-managed stop_price.
    - TP is strategy-managed: on TP condition, reduce target_qty by tp_sell_fraction.
    - On successful TP sell fill, reserve a fraction of actual proceeds into reserved_cash.
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.cfg = config or DEFAULT_CONFIG

        # Strategy-owned: stablecoin cash held back from redeployment ("out of market")
        self.reserved_cash = 0.0

        # Bar counter for simple "every N days" scheduling
        self.bar_i = 0

        # Reserve bookkeeping: apply reserve when we see the sell fill in ctx.last_exec
        self._awaiting_reserve = False
        self._awaiting_reserve_frac = 0.0

    def _p(self, key: str, default):
        params = getattr(self.cfg, "params", {}) or {}
        return params.get(key, default)

    def _get_ema(self, ctx: StrategyContext, n: int) -> float:
        v = ctx.features.get(f"ema_{int(n)}")
        return float(v) if v is not None else float("nan")

    def _get_rsi(self, ctx: StrategyContext) -> float:
        v = ctx.features.get("rsi_14")
        return float(v) if v is not None else float("nan")

    def _buy_filter_ok(self, ctx: StrategyContext, price: float) -> bool:
        f = str(self._p("buy_filter", "none")).strip().lower()
        if f in {"none", ""}:
            return True

        if f == "below_ema":
            ema_len = int(self._p("ema_len", 200) or 200)
            ema = self._get_ema(ctx, ema_len)
            return math.isfinite(ema) and price < ema

        if f == "rsi_below":
            thr = float(self._p("rsi_thr", 40.0) or 40.0)
            rsi = self._get_rsi(ctx)
            return math.isfinite(rsi) and rsi < thr

        return False

    def _deposit_due(self) -> bool:
        n = _freq_to_days(str(self._p("deposit_freq", "none")))
        if n <= 0:
            return False
        return (self.bar_i % n) == 0

    def _buy_due(self) -> bool:
        n = _freq_to_days(str(self._p("buy_freq", "weekly")))
        if n <= 0:
            return False
        return (self.bar_i % n) == 0

    def _tp_hit(self, ctx: StrategyContext) -> Tuple[bool, float]:
        tp_pct = float(self._p("tp_pct", 0.0) or 0.0)
        if tp_pct <= 0:
            return False, 0.0
        if abs(ctx.position.qty) <= 1e-12 or ctx.position.avg_entry <= 0:
            return False, 0.0

        tp_price = float(ctx.position.avg_entry) * (1.0 + float(tp_pct))
        hit = float(ctx.candle.high) >= tp_price
        return bool(hit), float(tp_price)

    def _stop_price(self, ctx: StrategyContext) -> Optional[float]:
        sl_pct = float(self._p("sl_pct", 0.0) or 0.0)
        if sl_pct <= 0:
            return None
        if abs(ctx.position.qty) <= 1e-12 or ctx.position.avg_entry <= 0:
            return None
        return float(ctx.position.avg_entry) * (1.0 - float(sl_pct))

    def _apply_post_sell_reserve_if_any(self, ctx: StrategyContext) -> None:
        """
        If we previously scheduled a TP sell and it filled at bar open, reserve a fraction
        of the actual proceeds (qty_sold * fill_price).
        """
        if not self._awaiting_reserve:
            return

        ex = ctx.last_exec
        if ex is None:
            return

        # If it was attempted and failed, cancel reserve (no proceeds).
        if ex.attempted and (not ex.filled):
            self._awaiting_reserve = False
            self._awaiting_reserve_frac = 0.0
            return

        if not ex.filled:
            return

        if ex.delta_qty >= -1e-12:
            # Not a sell fill
            self._awaiting_reserve = False
            self._awaiting_reserve_frac = 0.0
            return

        proceeds = abs(float(ex.delta_qty)) * float(ex.fill_price)
        frac = float(self._awaiting_reserve_frac)
        reserve_add = proceeds * frac

        # Clamp reserve to available cash (never exceed actual cash)
        cash = float(ctx.account.cash)
        self.reserved_cash = min(float(cash), float(self.reserved_cash + reserve_add))

        self._awaiting_reserve = False
        self._awaiting_reserve_frac = 0.0

    def on_bar(self, ctx: StrategyContext) -> Optional[PlanUpdate]:
        # First: if a TP sell happened at this bar open, reserve proceeds.
        self._apply_post_sell_reserve_if_any(ctx)

        # Keep reserved_cash sane if cash drops (fees, spends, etc.)
        self.reserved_cash = min(self.reserved_cash, float(ctx.account.cash))

        price = float(ctx.candle.close)
        if (not math.isfinite(price)) or price <= 0:
            self.bar_i += 1
            return PlanUpdate(action=PlanAction.HOLD)

        pos_qty = float(ctx.position.qty)
        equity = float(ctx.account.equity)
        cash_now = float(ctx.account.cash)

        # Deposits always happen when due (applied next bar open).
        cash_delta = 0.0
        if self._deposit_due():
            cash_delta = float(self._p("deposit_amount_usd", 0.0) or 0.0)
            if (not math.isfinite(cash_delta)) or cash_delta < 0:
                cash_delta = 0.0

        # Forecast deployable cash at next open (deposit is applied before trade).
        deployable_cash = max(0.0, cash_now - self.reserved_cash) + cash_delta

        # Allocation cap
        max_alloc = float(self._p("max_alloc_pct", 1.0) or 1.0)
        max_alloc = _clamp(max_alloc, 0.0, 1.0)
        invested_usd = max(0.0, pos_qty) * price
        cap_usd = equity * max_alloc if equity > 0 else 0.0
        remaining_cap_usd = max(0.0, cap_usd - invested_usd)

        # TP sell logic (strategy-managed)
        tp_hit, _tp_price = self._tp_hit(ctx)
        tp_sell_fraction = float(self._p("tp_sell_fraction", 1.0) or 1.0)
        tp_sell_fraction = _clamp(tp_sell_fraction, 0.0, 1.0)
        reserve_frac = float(self._p("reserve_frac_of_proceeds", 0.0) or 0.0)
        reserve_frac = _clamp(reserve_frac, 0.0, 1.0)

        target_qty = pos_qty
        will_sell = False
        if tp_hit and pos_qty > 1e-12 and tp_sell_fraction > 0:
            target_qty = pos_qty * (1.0 - tp_sell_fraction)
            will_sell = (target_qty < pos_qty - 1e-12)

        # Buy logic (DCA)
        will_buy = False
        if (not will_sell) and self._buy_due():
            if self._buy_filter_ok(ctx, price):
                buy_amt = float(self._p("buy_amount_usd", 0.0) or 0.0)
                if math.isfinite(buy_amt) and buy_amt > 0:
                    buy_usd = min(buy_amt, deployable_cash, remaining_cap_usd)
                    if buy_usd > 1e-9:
                        add_qty = buy_usd / price
                        target_qty = pos_qty + add_qty
                        will_buy = True

        # If nothing to do (no deposit, no buy, no sell), HOLD.
        if abs(cash_delta) < 1e-12 and (not will_buy) and (not will_sell):
            self.bar_i += 1
            return PlanUpdate(action=PlanAction.HOLD)

        # SL remains active (engine-managed stop)
        stop_price = self._stop_price(ctx)

        plan = TradePlan(
            desired_side=1,
            target_qty=float(max(0.0, target_qty)),
            entry_order_type=OrderType.MARKET,
            stop_price=float(stop_price) if stop_price is not None else None,
            take_profits=[],
            cash_delta=float(cash_delta),
            metadata={
                "strategy": str(self.cfg.strategy_name),
                "reserved_cash": float(self.reserved_cash),
                "deposit_due": bool(self._deposit_due()),
                "buy_due": bool(self._buy_due()),
                "tp_hit": bool(tp_hit),
            },
        )

        # If we are issuing a TP sell, arm reserve logic for next bar (after fill).
        if will_sell and reserve_frac > 0:
            self._awaiting_reserve = True
            self._awaiting_reserve_frac = float(reserve_frac)

        self.bar_i += 1
        return PlanUpdate(action=PlanAction.REPLACE, plan=plan)


# Convenience alias for CLI usage: --strategy strategies.dca_swing:Strategy
Strategy = DCASwingStrategy