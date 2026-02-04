from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

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
    return 0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _is_finite(x: Any) -> bool:
    try:
        return bool(math.isfinite(float(x)))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Default config (spot daily DCA + swing overlay)
# ---------------------------------------------------------------------------

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

        # Legacy single-filter (kept for backward compatibility)
        "buy_filter": "none",  # none|below_ema|rsi_below|macd_bull|bb_z_below|adx_above|donch_pos_below
        "ema_len": 200,
        "rsi_thr": 40.0,
        "macd_hist_thr": 0.0,
        "bb_z_thr": -1.0,
        "adx_thr": 20.0,
        "donch_pos_thr": 0.20,

        # New: structured entry logic (preferred).
        # Schema: {"regime":[cond,...], "clauses":[[cond,...], ...]}
        # cond schema mirrors make_dca_grid_strategy_overhaul_v1.py:
        #   {"indicator":"close","operator":"<","threshold":0.0,"ref_indicator":"ema_200"}
        "entry_logic": {"regime": [], "clauses": []},

        # Allocation cap
        "max_alloc_pct": 1.0,

        # Risk controls (fractions: 0.10 = 10%)
        "sl_pct": 0.0,      # hard SL from avg_entry*(1-sl_pct); 0 disables
        "trail_pct": 0.0,   # trailing stop from peak*(1-trail_pct); 0 disables
        "max_hold_bars": 0, # time stop (bars since position opened); 0 disables

        # Take profit (strategy-managed)
        "tp_pct": 0.0,  # avg_entry*(1+tp_pct); 0 disables
        "tp_sell_fraction": 0.50,
        "reserve_frac_of_proceeds": 0.50,
    },
)


# ---------------------------------------------------------------------------
# Entry logic evaluation (v1): regime gate + OR-of-AND triggers
# ---------------------------------------------------------------------------

_ALLOWED_OPS = {"<", "<=", ">", ">="}


def _get_indicator(ctx: StrategyContext, name: str) -> float:
    """
    Fetch indicator values safely.
    - OHLC: open/high/low/close from ctx.candle
    - Other: from ctx.features (dict-like via .get)
    """
    k = str(name or "").strip()
    if not k:
        return float("nan")

    lk = k.lower()
    c = ctx.candle
    if lk == "open":
        return float(c.open)
    if lk == "high":
        return float(c.high)
    if lk == "low":
        return float(c.low)
    if lk == "close":
        return float(c.close)
    if lk in {"volume", "vol"}:
        v = getattr(c, "volume", None)
        return float(v) if v is not None else float("nan")

    v2 = ctx.features.get(k)
    return float(v2) if v2 is not None else float("nan")


def _eval_condition(ctx: StrategyContext, cond: Dict[str, Any]) -> bool:
    """
    Condition schema (supported keys):
      - indicator (lhs)
      - operator: one of <, <=, >, >=
      - threshold: float (rhs value) OR float offset when ref_indicator exists
      - ref_indicator: optional feature name for rhs
    """
    if not isinstance(cond, dict):
        return False

    ind = cond.get("indicator") or cond.get("feature") or cond.get("lhs")
    op = cond.get("operator") or cond.get("op")
    thr = cond.get("threshold", cond.get("value", 0.0))
    ref = cond.get("ref_indicator") or cond.get("rhs") or cond.get("rhs_indicator")

    op = str(op or "").strip()
    if op not in _ALLOWED_OPS:
        return False

    lhs = _get_indicator(ctx, str(ind or ""))
    if not _is_finite(lhs):
        return False

    # rhs: either a ref indicator (plus optional offset) or a literal threshold
    rhs: float
    if ref is not None and str(ref).strip():
        rhs0 = _get_indicator(ctx, str(ref))
        if not _is_finite(rhs0):
            return False
        off = float(thr or 0.0)
        rhs = float(rhs0) + off
    else:
        rhs = float(thr)

    if not _is_finite(rhs):
        return False

    if op == "<":
        return lhs < rhs
    if op == "<=":
        return lhs <= rhs
    if op == ">":
        return lhs > rhs
    if op == ">=":
        return lhs >= rhs
    return False


def _normalize_entry_logic(entry_logic: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(entry_logic, dict):
        return None

    regime = entry_logic.get("regime", [])
    clauses = entry_logic.get("clauses", [])

    if not isinstance(regime, list):
        regime = []
    if not isinstance(clauses, list):
        clauses = []

    regime2 = [c for c in regime if isinstance(c, dict)]
    clauses2: List[List[Dict[str, Any]]] = []
    for cl in clauses:
        if isinstance(cl, list):
            clauses2.append([c for c in cl if isinstance(c, dict)])

    return {"regime": regime2, "clauses": clauses2}


def _entry_logic_ok(ctx: StrategyContext, entry_logic: Dict[str, Any]) -> bool:
    # Regime gate: AND
    for c in entry_logic.get("regime", []) or []:
        if not _eval_condition(ctx, c):
            return False

    clauses = entry_logic.get("clauses", []) or []
    if not clauses:
        # No triggers means "always allowed" (useful for buy_filter="none")
        return True

    # OR-of-AND
    for clause in clauses:
        if not clause:
            return True
        ok = True
        for c in clause:
            if not _eval_condition(ctx, c):
                ok = False
                break
        if ok:
            return True

    return False


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class DCASwingStrategy:
    """
    Daily-bar DCA + Swing overlay template.

    Additions (v1 strategy overhaul):
    - Entry logic: regime gate + OR-of-AND triggers (entry_logic param).
    - Time stop: flatten after max_hold_bars.
    - Trailing stop: peak*(1-trail_pct), ratcheting only upward.
    - Receipts-friendly tagging:
        - plan.metadata["exit_tag"]="time" when time stop triggers
        - plan.metadata["stop_kind"]="TRAIL" vs "SL" when a stop is armed
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.cfg = config or DEFAULT_CONFIG

        # Stablecoin cash held back from redeployment ("out of market")
        self.reserved_cash = 0.0

        # Scheduling counter
        self.bar_i = 0

        # Reserve bookkeeping: apply reserve when we see the TP sell fill in ctx.last_exec
        self._awaiting_reserve = False
        self._awaiting_reserve_frac = 0.0

        # Position lifecycle state (observed at bar close, after any fills at bar open)
        self._prev_pos_qty = 0.0
        self._pos_open_bar_i: Optional[int] = None
        self._peak_since_open: float = float("nan")

    def _p(self, key: str, default):
        params = getattr(self.cfg, "params", {}) or {}
        return params.get(key, default)

    # ----------------------
    # Legacy buy filters (kept)
    # ----------------------
    def _get_ema(self, ctx: StrategyContext, n: int) -> float:
        v = ctx.features.get(f"ema_{int(n)}")
        return float(v) if v is not None else float("nan")

    def _get_macd_hist(self, ctx: StrategyContext) -> float:
        v = ctx.features.get("macd_hist_12_26_9")
        return float(v) if v is not None else float("nan")

    def _get_bb_z(self, ctx: StrategyContext) -> float:
        v = ctx.features.get("bb_z_20")
        return float(v) if v is not None else float("nan")

    def _get_adx(self, ctx: StrategyContext) -> float:
        v = ctx.features.get("adx_14")
        return float(v) if v is not None else float("nan")

    def _get_donch_pos(self, ctx: StrategyContext) -> float:
        v = ctx.features.get("donch_pos_20")
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
            return math.isfinite(rsi) and rsi <= thr

        if f == "macd_bull":
            thr = float(self._p("macd_hist_thr", 0.0) or 0.0)
            h = self._get_macd_hist(ctx)
            return math.isfinite(h) and h >= thr

        if f == "bb_z_below":
            thr = float(self._p("bb_z_thr", -1.0) or -1.0)
            z = self._get_bb_z(ctx)
            return math.isfinite(z) and z <= thr

        if f == "adx_above":
            thr = float(self._p("adx_thr", 20.0) or 20.0)
            adx = self._get_adx(ctx)
            return math.isfinite(adx) and adx >= thr

        if f == "donch_pos_below":
            thr = float(self._p("donch_pos_thr", 0.20) or 0.20)
            pos = self._get_donch_pos(ctx)
            return math.isfinite(pos) and pos <= thr

        return False

    # ----------------------
    # Scheduling
    # ----------------------
    def _deposit_due(self) -> bool:
        n = _freq_to_days(str(self._p("deposit_freq", "none")))
        return (n > 0) and ((self.bar_i % n) == 0)

    def _buy_due(self) -> bool:
        n = _freq_to_days(str(self._p("buy_freq", "weekly")))
        return (n > 0) and ((self.bar_i % n) == 0)

    # ----------------------
    # TP / SL helpers
    # ----------------------
    def _tp_hit(self, ctx: StrategyContext) -> Tuple[bool, float]:
        tp_pct = float(self._p("tp_pct", 0.0) or 0.0)
        if tp_pct <= 0:
            return False, 0.0
        if abs(ctx.position.qty) <= 1e-12 or ctx.position.avg_entry <= 0:
            return False, 0.0
        tp_price = float(ctx.position.avg_entry) * (1.0 + float(tp_pct))
        hit = float(ctx.candle.high) >= tp_price
        return bool(hit), float(tp_price)

    def _sl_stop(self, ctx: StrategyContext) -> Optional[float]:
        sl_pct = float(self._p("sl_pct", 0.0) or 0.0)
        if sl_pct <= 0:
            return None
        if abs(ctx.position.qty) <= 1e-12 or ctx.position.avg_entry <= 0:
            return None
        return float(ctx.position.avg_entry) * (1.0 - float(sl_pct))

    def _trail_stop(self, ctx: StrategyContext) -> Optional[float]:
        trail_pct = float(self._p("trail_pct", 0.0) or 0.0)
        if trail_pct <= 0:
            return None
        if abs(ctx.position.qty) <= 1e-12:
            return None
        if not _is_finite(self._peak_since_open):
            return None
        return float(self._peak_since_open) * (1.0 - float(trail_pct))

    def _apply_post_sell_reserve_if_any(self, ctx: StrategyContext) -> None:
        if not self._awaiting_reserve:
            return
        ex = ctx.last_exec
        if ex is None:
            return

        if ex.attempted and (not ex.filled):
            self._awaiting_reserve = False
            self._awaiting_reserve_frac = 0.0
            return

        if not ex.filled:
            return

        if ex.delta_qty >= -1e-12:
            self._awaiting_reserve = False
            self._awaiting_reserve_frac = 0.0
            return

        proceeds = abs(float(ex.delta_qty)) * float(ex.fill_price)
        frac = float(self._awaiting_reserve_frac)
        reserve_add = proceeds * frac

        cash = float(ctx.account.cash)
        self.reserved_cash = min(float(cash), float(self.reserved_cash + reserve_add))

        self._awaiting_reserve = False
        self._awaiting_reserve_frac = 0.0

    # ----------------------
    # Core loop
    # ----------------------
    def on_bar(self, ctx: StrategyContext) -> Optional[PlanUpdate]:
        # Apply reserve after a TP sell fill (if any)
        self._apply_post_sell_reserve_if_any(ctx)

        # Keep reserved_cash sane if cash drops (fees, spends, etc.)
        self.reserved_cash = min(self.reserved_cash, float(ctx.account.cash))

        price = float(ctx.candle.close)
        if (not math.isfinite(price)) or price <= 0:
            self.bar_i += 1
            self._prev_pos_qty = float(ctx.position.qty)
            return PlanUpdate(action=PlanAction.HOLD)

        pos_qty = float(ctx.position.qty)
        equity = float(ctx.account.equity)
        cash_now = float(ctx.account.cash)

        # Detect position open/close events (observed state at bar close)
        if self._prev_pos_qty <= 1e-12 and pos_qty > 1e-12:
            self._pos_open_bar_i = int(self.bar_i)
            self._peak_since_open = float(ctx.candle.high) if _is_finite(ctx.candle.high) else price
        elif pos_qty <= 1e-12:
            self._pos_open_bar_i = None
            self._peak_since_open = float("nan")

        # Update peak while in position
        if pos_qty > 1e-12:
            h = float(ctx.candle.high)
            if _is_finite(h):
                if not _is_finite(self._peak_since_open):
                    self._peak_since_open = h
                else:
                    self._peak_since_open = max(float(self._peak_since_open), h)

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

        # ---------
        # Exit management: time stop takes precedence over TP/buys
        # ---------
        max_hold_bars = int(self._p("max_hold_bars", 0) or 0)
        time_stop_due = False
        if max_hold_bars > 0 and pos_qty > 1e-12 and self._pos_open_bar_i is not None:
            held = int(self.bar_i) - int(self._pos_open_bar_i)
            if held >= int(max_hold_bars):
                time_stop_due = True

        # TP sell logic (strategy-managed partial reduction)
        tp_hit, _tp_price = self._tp_hit(ctx)
        tp_sell_fraction = float(self._p("tp_sell_fraction", 1.0) or 1.0)
        tp_sell_fraction = _clamp(tp_sell_fraction, 0.0, 1.0)
        reserve_frac = float(self._p("reserve_frac_of_proceeds", 0.0) or 0.0)
        reserve_frac = _clamp(reserve_frac, 0.0, 1.0)

        target_qty = pos_qty
        will_sell = False
        exit_tag: Optional[str] = None

        if time_stop_due:
            target_qty = 0.0
            will_sell = (pos_qty > 1e-12)
            exit_tag = "time"
        else:
            if tp_hit and pos_qty > 1e-12 and tp_sell_fraction > 0:
                target_qty = pos_qty * (1.0 - tp_sell_fraction)
                will_sell = (target_qty < pos_qty - 1e-12)

        # ---------
        # Buy logic (DCA)
        # ---------
        will_buy = False
        if (not will_sell) and self._buy_due():
            entry_logic = _normalize_entry_logic(self._p("entry_logic", None))
            if entry_logic is not None:
                ok = _entry_logic_ok(ctx, entry_logic)
            else:
                ok = self._buy_filter_ok(ctx, price)

            if ok:
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
            self._prev_pos_qty = float(pos_qty)
            return PlanUpdate(action=PlanAction.HOLD)

        # ---------
        # Stops: hard SL + trailing stop (ratcheting stop uses max)
        # ---------
        sl_stop = self._sl_stop(ctx)
        tr_stop = self._trail_stop(ctx)
        stop_price: Optional[float] = None
        stop_kind: Optional[str] = None

        if (sl_stop is not None) or (tr_stop is not None):
            if sl_stop is None:
                stop_price = float(tr_stop) if tr_stop is not None else None
                stop_kind = "TRAIL"
            elif tr_stop is None:
                stop_price = float(sl_stop)
                stop_kind = "SL"
            else:
                # Choose tighter stop (higher price for long)
                stop_price = float(max(float(sl_stop), float(tr_stop)))
                # If trailing is the active bound, label as TRAIL
                stop_kind = "TRAIL" if abs(stop_price - float(tr_stop)) <= 1e-9 and float(tr_stop) >= float(sl_stop) - 1e-9 else "SL"

        # If we are forcing a time stop exit (target_qty == 0), disable stop.
        if exit_tag is not None and float(target_qty) <= 1e-12:
            stop_price = None
            stop_kind = None

        meta = {
            "strategy": str(self.cfg.strategy_name),
            "reserved_cash": float(self.reserved_cash),
            "deposit_due": bool(self._deposit_due()),
            "buy_due": bool(self._buy_due()),
            "tp_hit": bool(tp_hit),
            "pos_open_bar_i": int(self._pos_open_bar_i) if self._pos_open_bar_i is not None else None,
        }
        if exit_tag is not None:
            meta["exit_tag"] = str(exit_tag)
        if stop_kind is not None:
            meta["stop_kind"] = str(stop_kind)
        if _is_finite(self._peak_since_open):
            meta["peak_since_open"] = float(self._peak_since_open)

        plan = TradePlan(
            desired_side=1,
            target_qty=float(max(0.0, target_qty)),
            entry_order_type=OrderType.MARKET,
            stop_price=float(stop_price) if stop_price is not None else None,
            take_profits=[],
            cash_delta=float(cash_delta),
            metadata=meta,
        )

        # If we are issuing a TP sell, arm reserve logic for next bar (after fill).
        if will_sell and (exit_tag is None) and reserve_frac > 0:
            self._awaiting_reserve = True
            self._awaiting_reserve_frac = float(reserve_frac)

        self.bar_i += 1
        self._prev_pos_qty = float(pos_qty)
        return PlanUpdate(action=PlanAction.REPLACE, plan=plan)


# Convenience alias for CLI usage
Strategy = DCASwingStrategy
