# strategies/universal.py
import math
from typing import Any, Optional, Tuple

from engine.contracts import (
    EntryCondition,
    FilterCondition,
    GuardrailsSpec,
    OrderType,
    PlanAction,
    PlanUpdate,
    RiskConfig,
    StrategyConfig,
    StrategyContext,
    TakeProfitLevel,
    TradePlan,
)

DEFAULT_CONFIG = StrategyConfig(
    strategy_name="Universal_Demo_Trend",
    side="long",
    entry_conditions=[
        EntryCondition(indicator="rsi_14", operator="<", threshold=45.0),
    ],
    filters=[
        FilterCondition(
            indicator="close",
            operator=">",
            threshold=0.0,
            ref_indicator="ema_200",
        ),
        FilterCondition(
            indicator="close",
            operator=">",
            threshold=0.0,
            ref_indicator="ema_50",
        ),
    ],
    risk=RiskConfig(
        risk_per_trade_pct=0.01,
        max_leverage=5.0,
        sl_type="ATR",
        sl_param=2.5,
        tp_r_multiples=[2.0, 5.0],
        tp_fractions=[0.5, 1.0],
        move_to_be_at_r=2.0,
    ),
    guardrails=GuardrailsSpec(
        max_daily_loss_pct=0.05,
        cooldown_base_min=60,
    ),
)


class UniversalStrategy:
    def __init__(self, config: Optional[StrategyConfig] = None):
        self.cfg = config if config else DEFAULT_CONFIG
        self._side_mult = 1 if self.cfg.side == "long" else -1

    def _get_val(self, ctx: StrategyContext, indicator: str) -> float:
        # Candle fields
        if indicator in ["close", "open", "high", "low", "volume"]:
            v = getattr(ctx.candle, indicator)
            return float(v) if v is not None else float("nan")

        # Feature fields
        v = ctx.features.get(indicator)
        return float(v) if v is not None else float("nan")

    def _clamp01(self, x: float) -> float:
        if not math.isfinite(x):
            return 0.0
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    def _lin_norm(self, x: float, lo: float, hi: float) -> float:
        if (not math.isfinite(x)) or (not math.isfinite(lo)) or (not math.isfinite(hi)):
            return 0.0
        if hi <= lo:
            return 0.0
        return self._clamp01((x - lo) / (hi - lo))
        
    def _confidence_u(self, ctx: StrategyContext) -> float:
        r = self.cfg.risk
        adx = float(ctx.features.get("adx_14_1h", float("nan")))
        bbw = float(ctx.features.get("bb_width_20_1h", float("nan")))
        rvol = float(ctx.features.get("rvol_50", float("nan")))

        u_adx = self._lin_norm(adx, float(r.conf_adx_lo), float(r.conf_adx_hi))
        u_rvol = self._lin_norm(rvol, float(r.conf_rvol_lo), float(r.conf_rvol_hi))

        if str(getattr(r, "conf_model", "compression")).lower() == "expansion":
            u_bbw = self._lin_norm(bbw, float(r.conf_bbw_lo), float(r.conf_bbw_hi))
        else:
            # compression: lower bb width => higher confidence
            u_bbw = 1.0 - self._lin_norm(bbw, float(r.conf_bbw_lo), float(r.conf_bbw_hi))

        u = (
            float(r.conf_w_adx) * u_adx
            + float(r.conf_w_bbw) * u_bbw
            + float(r.conf_w_rvol) * u_rvol
        )
        return self._clamp01(u)

    def _check(self, ctx: StrategyContext, cond: Any) -> bool:
        val_left = self._get_val(ctx, cond.indicator)

        val_right = float(cond.threshold)
        if getattr(cond, "ref_indicator", None):
            ref = self._get_val(ctx, cond.ref_indicator)
            val_right = val_right + ref

        # If anything is NaN/inf, treat as "condition not met"
        if (not math.isfinite(val_left)) or (not math.isfinite(val_right)):
            return False

        op = cond.operator
        if op == ">":
            return val_left > val_right
        if op == "<":
            return val_left < val_right
        if op == ">=":
            return val_left >= val_right
        if op == "<=":
            return val_left <= val_right
        return False

    def _entry_passes(self, ctx: StrategyContext) -> Tuple[bool, Optional[int]]:
        """
        Entry logic (backward compatible):
          - Option 2: entry_any = OR of AND-clauses
          - Legacy: entry_conditions = AND list

        Returns: (passes, triggered_clause_index)
          - triggered_clause_index is only meaningful when entry_any is used.
        """
        entry_any = getattr(self.cfg, "entry_any", None)
        if entry_any:
            for ci, clause in enumerate(entry_any):
                # Clause passes if ALL conditions pass
                ok = True
                for cond in clause:
                    if not self._check(ctx, cond):
                        ok = False
                        break
                if ok:
                    return True, int(ci)
            return False, None

        # Legacy: ALL entry_conditions must pass
        for c in self.cfg.entry_conditions:
            if not self._check(ctx, c):
                return False, None
        return True, None

    def _hard_trend_gate_ok(self, ctx: StrategyContext) -> bool:
       """
       Belt-and-suspenders: require higher timeframe trend direction if available.
       Backward compatible: if feature is missing/NaN, do not block.
       """
       if self.cfg.side == "long":
           v = ctx.features.get("trend_up_1h")
           if v is None:
               return True
           try:
               x = float(v)
               return (math.isfinite(x) and x > 0.5) or (not math.isfinite(x))
           except Exception:
               return True

       v = ctx.features.get("trend_down_1h")
       if v is None:
           return True
       try:
           x = float(v)
           return (math.isfinite(x) and x > 0.5) or (not math.isfinite(x))
       except Exception:
           return True

   

    def on_bar(self, ctx: StrategyContext) -> Optional[PlanUpdate]:
        # If we're in a position, let the engine manage TP/SL.
        if abs(ctx.position.qty) > 1e-12:
            return PlanUpdate(action=PlanAction.HOLD)

        # Production safety gate: only trade with higher timeframe trend direction.
        if not self._hard_trend_gate_ok(ctx):
           return PlanUpdate(action=PlanAction.HOLD)


        # Filters (AND)
        for f in self.cfg.filters:
            if not self._check(ctx, f):
                return PlanUpdate(action=PlanAction.HOLD)

        # Entry logic (Option 2 OR-of-AND, else legacy AND)
        entry_ok, trig_clause = self._entry_passes(ctx)
        if not entry_ok:
            return PlanUpdate(action=PlanAction.HOLD)

        risk = self.cfg.risk
        price = float(ctx.candle.close)

        if (not math.isfinite(price)) or price <= 0:
            return PlanUpdate(action=PlanAction.HOLD)

        # Stop distance
        dist = 0.0
        if risk.sl_type == "ATR":
            atr = ctx.features.get("atr_14")
            if atr is None:
                return PlanUpdate(action=PlanAction.HOLD)
            atr = float(atr)
            if (not math.isfinite(atr)) or atr <= 0:
                return PlanUpdate(action=PlanAction.HOLD)
            dist = atr * float(risk.sl_param)

        elif risk.sl_type == "PCT":
            dist = price * float(risk.sl_param)

        else:
            # Unknown stop model
            return PlanUpdate(action=PlanAction.HOLD)

        if (not math.isfinite(dist)) or dist <= 0:
            return PlanUpdate(action=PlanAction.HOLD)

        equity = float(ctx.account.equity)
        if (not math.isfinite(equity)) or equity <= 0:
            return PlanUpdate(action=PlanAction.HOLD)

        # Sizing v2: dynamic leverage (full wallet collateral)
        sizing_mode = str(getattr(risk, "sizing_mode", "legacy")).lower()
        if sizing_mode == "dynamic_leverage":
            s_model = float(dist / price) if price > 0 else float("nan")
            if (not math.isfinite(s_model)) or s_model <= 0:
                return PlanUpdate(action=PlanAction.HOLD)

            stop_floor = float(getattr(risk, "stop_floor_pct", 0.003))
            cost_frac = float(getattr(risk, "cost_bps_assumption", 12.0)) / 10_000.0
            s_eff = max(float(s_model), float(stop_floor)) + float(cost_frac)

            # Allowed leverage band
            L_lo = float(getattr(risk, "leverage_min", 2.0))
            L_hi = float(getattr(risk, "leverage_max", 15.0))

            # Cap by engine constraint
            L_hi = min(L_hi, float(ctx.constraints.max_leverage))
        
            r_max = float(getattr(risk, "risk_max_pct", 0.05))
            L_risk_max = r_max / s_eff if s_eff > 0 else 0.0

            # Liquidation safety cap
            mmr = float(ctx.constraints.maint_margin_rate)
            beta = float(getattr(risk, "liq_safety_frac", 0.20))
            beta = 0.0 if beta < 0 else (0.90 if beta > 0.90 else beta)
            denom = mmr + (float(s_model) / max(1e-9, (1.0 - beta)))
            L_liq_max = (1.0 / denom) if denom > 1e-12 else 0.0
        
            L_hi = min(L_hi, L_risk_max, L_liq_max)
            if (not math.isfinite(L_hi)) or L_hi < L_lo:
                return PlanUpdate(action=PlanAction.HOLD)

            u = self._confidence_u(ctx)
            L = float(L_lo + u * (L_hi - L_lo))
            if (not math.isfinite(L)) or L <= 0:
                return PlanUpdate(action=PlanAction.HOLD)

            notional = equity * L
            qty_abs = notional / price

            # Risk dollars (approx) at stop including cost cushion
            risk_amt = equity * L * s_eff
        else:
            # Legacy sizing: fixed loss-at-stop % of equity
            risk_amt = equity * float(risk.risk_per_trade_pct)
            if (not math.isfinite(risk_amt)) or risk_amt <= 0:
                return PlanUpdate(action=PlanAction.HOLD)
            qty_abs = risk_amt / dist
            if (not math.isfinite(qty_abs)) or qty_abs <= 0:
                return PlanUpdate(action=PlanAction.HOLD)
            max_qty_abs = (equity * float(risk.max_leverage)) / price
            if (not math.isfinite(max_qty_abs)) or max_qty_abs <= 0:
                return PlanUpdate(action=PlanAction.HOLD)
            qty_abs = min(qty_abs, max_qty_abs)

        qty = float(self._side_mult) * float(qty_abs)

        if (not math.isfinite(qty)) or abs(qty) <= 0:
            return PlanUpdate(action=PlanAction.HOLD)

        stop_price = price - (dist * float(self._side_mult))

        # TP ladder
        tps = []
        tp_order_type = (
            OrderType.MARKET if bool(getattr(risk, "tp_is_market", False)) else OrderType.LIMIT
        )
        for r_mult, frac in zip(risk.tp_r_multiples, risk.tp_fractions):
            r_mult = float(r_mult)
            frac = float(frac)

            if (not math.isfinite(r_mult)) or r_mult <= 0:
                continue
            if (not math.isfinite(frac)) or frac <= 0:
                continue

            tp_price = price + (dist * r_mult * float(self._side_mult))

            move_sl = None
            if risk.move_to_be_at_r is not None:
                be_at = float(risk.move_to_be_at_r)
                if math.isfinite(be_at) and r_mult >= be_at:
                    move_sl = price  # breakeven

            tps.append(
                TakeProfitLevel(
                    price=float(tp_price),
                    fraction_of_initial=float(frac),
                    order_type=tp_order_type,
                    move_stop_to=float(move_sl) if move_sl is not None else None,
                )
            )

        # Metadata: keep it small but useful
        meta = {"strategy": self.cfg.strategy_name, "risk_usd": float(risk_amt)}
        if risk.sl_type == "ATR":
            meta["atr_entry"] = float(ctx.features.get("atr_14", 0.0))

        if getattr(self.cfg, "entry_any", None):
            # Helpful for debugging which OR-clause fired the entry.
            meta["triggered_clause_index"] = int(trig_clause) if trig_clause is not None else None    

        plan = TradePlan(
            desired_side=int(self._side_mult),
            target_qty=float(qty),
            entry_order_type=OrderType.MARKET,
            stop_price=float(stop_price),
            take_profits=tps,
            guardrails=self.cfg.guardrails,
            metadata=meta,
        )

        return PlanUpdate(action=PlanAction.REPLACE, plan=plan)