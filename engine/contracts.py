# engine/contracts.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class Candle:
    dt: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    vol_bps: float = 0.0
    liq_mult: float = 1.0


@dataclass
class AccountState:
    cash: float
    equity: float
    realized_pnl_gross: float
    fees_paid_total: float
    funding_net: float


@dataclass
class PositionState:
    qty: float
    avg_entry: float
    unrealized_pnl: float
    liq_price: Optional[float]


@dataclass
class EngineConstraints:
    price_tick: float
    qty_step: float
    min_notional_usdt: float
    max_leverage: float
    maint_margin_rate: float


@dataclass
class GuardrailsSpec:
    min_equity_stop: Optional[float] = None
    max_daily_loss_pct: Optional[float] = None
    close_on_daily_limit: bool = False
    base_cooldown_minutes: int = 0
    loss_streak_start: int = 0
    cooldown_base_min: int = 15
    cooldown_max_min: int = 24 * 60


@dataclass
class ExitPolicy:
    allow_flip: bool = True


@dataclass
class TakeProfitLevel:
    price: float
    fraction_of_initial: float
    order_type: OrderType = OrderType.LIMIT
    move_stop_to: Optional[float] = None


@dataclass
class TradePlan:
    desired_side: int  # -1 short, 0 flat, +1 long
    target_qty: Optional[float] = None
    target_notional: Optional[float] = None
    entry_order_type: OrderType = OrderType.MARKET
    entry_limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    take_profits: List[TakeProfitLevel] = field(default_factory=list)
    guardrails: Optional[GuardrailsSpec] = None
    exit_policy: Optional[ExitPolicy] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    expires_after_bars: Optional[int] = None


class PlanAction(str, Enum):
    HOLD = "hold"
    REPLACE = "replace"
    CANCEL = "cancel"


@dataclass
class PlanUpdate:
    action: PlanAction = PlanAction.HOLD
    plan: Optional[TradePlan] = None
    note: str = ""


@dataclass
class ExecutionResult:
    attempted: bool
    filled: bool
    reject_reason: Optional[str] = None
    leverage_clamped: bool = False
    qty_rounded_to_step: bool = False
    price_rounded_to_tick: bool = False
    delta_qty: float = 0.0
    fill_price: float = 0.0
    fee_paid: float = 0.0

class FeatureStore(Protocol):
    """
    Minimal interface for per-bar feature access.

    - dict satisfies this (has .get)
    - backtester.FeatureView satisfies this (has .get)
    """
    def get(self, key: str, default: Any = None) -> Any:
        ...


@dataclass
class StrategyContext:
    candle: Candle
    account: AccountState
    position: PositionState
    constraints: EngineConstraints
    # Feature store access for the current bar (dict-like via .get()).
    features: FeatureStore = field(default_factory=dict)
    last_exec: Optional[ExecutionResult] = None


# --- Strategy Factory Configuration Contracts ---

@dataclass
class EntryCondition:
    indicator: str      # e.g., "rsi_14", "close"
    operator: str       # "<", ">", ">=", "<="
    threshold: float    # Fixed value comparison
    ref_indicator: Optional[str] = None 


@dataclass
class FilterCondition:
    indicator: str
    operator: str
    threshold: float
    ref_indicator: Optional[str] = None


@dataclass
class RiskConfig:
    # Sizing
    risk_per_trade_pct: float = 0.01  # 1% equity risk
    max_leverage: float = 3.0

    # Stop Loss
    sl_type: str = "ATR"     # "ATR", "PCT"
    sl_param: float = 2.0    # 2.0x ATR or 0.02 (2%)

    # Take Profit Ladder
    tp_r_multiples: List[float] = field(default_factory=list)
    tp_fractions: List[float] = field(default_factory=list)
    
    # Management
    move_to_be_at_r: Optional[float] = None  # Move SL to Entry if price hits X*R profit

    # Execution
    tp_is_market: bool = False  # If True, TP ladder uses market orders (deterministic)

    # Sizing (v2)
    sizing_mode: str = "legacy"  # "legacy" or "dynamic_leverage"
    leverage_min: float = 2.0
    leverage_max: float = 15.0
    risk_max_pct: float = 0.05
    stop_floor_pct: float = 0.003
    liq_safety_frac: float = 0.20
    cost_bps_assumption: float = 12.0

    # Confidence model for leverage selection
    # "compression": lower bb_width => higher confidence
    # "expansion": higher bb_width => higher confidence
    conf_model: str = "compression"

    # Confidence mapping ranges (defaults are reasonable)
    conf_adx_lo: float = 15.0
    conf_adx_hi: float = 30.0
    conf_bbw_lo: float = 0.04
    conf_bbw_hi: float = 0.08
    conf_rvol_lo: float = 1.0
    conf_rvol_hi: float = 2.0
    conf_w_adx: float = 0.5
    conf_w_bbw: float = 0.3
    conf_w_rvol: float = 0.2


@dataclass
class StrategyConfig:
    strategy_name: str
    side: str  # "long", "short"
    
    # Legacy (v1): AND of conditions
    entry_conditions: List[EntryCondition] = field(default_factory=list)

    # Option 2 (bounded DNF-lite): OR of AND clauses.
    # - Outer list: OR clauses
    # - Inner list: AND conditions within a clause
    # Backward compatible: if None/empty, fall back to entry_conditions.
    entry_any: Optional[List[List[EntryCondition]]] = None

    filters: List[FilterCondition] = field(default_factory=list)
    
    risk: RiskConfig = field(default_factory=RiskConfig)
    guardrails: Optional[GuardrailsSpec] = None
