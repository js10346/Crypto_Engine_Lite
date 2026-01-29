# engine/batch.py
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import time
import os
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from engine.contracts import (
    EntryCondition,
    FilterCondition,
    GuardrailsSpec,
    RiskConfig,
    StrategyConfig,
)
from engine.features import add_features

from engine.backtester import (
    BacktestConfig,
    _build_cashflow_performance_stats,
    _build_performance_stats,
    _efficiency_stats,
    _execution_breakdown,
    _exit_counts,
    _exposure_stats,
    _stop_move_stats,
    _tp_level_counts,
    _tp_sl_profitability_report,
    _trade_stats,
    run_backtest_once,
)


class ConfigError(Exception):
    pass

# ============================================================
# Naming helpers (readable IDs)
# ============================================================


def _slug(s: str, *, max_len: int = 140) -> str:
    """
    Filesystem-friendly slug. Keeps letters, digits, underscore, dash, dot.
    """
    s = str(s or "").strip()
    if not s:
        return "run"
    s = s.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "run"
    return s[: int(max_len)]


def _freq_code(freq: Any) -> str:
    f = str(freq or "").strip().lower()
    if f in {"none", "off", "0", ""}:
        return "0"
    if f in {"daily", "day", "1d"}:
        return "D"
    if f in {"weekly", "week", "7d"}:
        return "W"
    if f in {"monthly", "month", "30d"}:
        return "M"
    return _slug(f, max_len=8)


def _pct100(x: Any) -> str:
    try:
        v = float(x)
        if not math.isfinite(v):
            return "na"
        return str(int(round(v * 100)))
    except Exception:
        return "na"


def _money0(x: Any) -> str:
    try:
        v = float(x)
        if not math.isfinite(v):
            return "na"
        return str(int(round(v)))
    except Exception:
        return "na"


def _dca_label(params: Dict[str, Any]) -> str:
    dep_f = _freq_code(params.get("deposit_freq", "none"))
    dep_a = _money0(params.get("deposit_amount_usd", 0.0))
    buy_f = _freq_code(params.get("buy_freq", "weekly"))
    buy_a = _money0(params.get("buy_amount_usd", 0.0))

    buy_filter = str(params.get("buy_filter", "none") or "none").strip().lower()
    if buy_filter == "below_ema":
        ema_len = int(params.get("ema_len", 200) or 200)
        filt = f"fEMA{ema_len}"
    elif buy_filter == "rsi_below":
        rsi_thr = params.get("rsi_thr", 40.0)
        filt = f"fRSI{_money0(rsi_thr)}"
    else:
        filt = "fNone"

    alloc = f"a{_pct100(params.get('max_alloc_pct', 1.0))}"
    sl = f"sl{_pct100(params.get('sl_pct', 0.0))}"
    tp = f"tp{_pct100(params.get('tp_pct', 0.0))}"
    sell = f"sell{_pct100(params.get('tp_sell_fraction', 1.0))}"
    res = f"res{_pct100(params.get('reserve_frac_of_proceeds', 0.0))}"

    # Full but still compact.
    return _slug(
        f"dep{dep_f}{dep_a}_buy{buy_f}{buy_a}_{filt}_{alloc}_{sl}_{tp}_{sell}_{res}",
        max_len=180,
    )


def _config_label(cfg: StrategyConfig) -> str:
    name = str(getattr(cfg, "strategy_name", "strategy") or "strategy").strip().lower()
    params = dict(getattr(cfg, "params", {}) or {})
    if name in {"dca", "dca_swing"}:
        return _dca_label(params)
    # Fallback: at least readable name + side.
    side = str(getattr(cfg, "side", "") or "").strip().lower()
    return _slug(f"{name}_{side}", max_len=80)



# ============================================================
# Multiprocessing globals (Windows spawn-safe)
# ============================================================

_WORKER_DF_FEAT: Optional[pd.DataFrame] = None
_WORKER_TEMPLATE_CLS: Any = None
_WORKER_CONSTRAINTS: Any = None
_WORKER_ENGINE_CFG: Optional[BacktestConfig] = None

# ============================================================
# Utilities
# ============================================================


def _import_symbol(dotted: str) -> Any:
    if ":" in dotted:
        mod_name, obj_name = dotted.split(":", 1)
    else:
        mod_name, obj_name = dotted, "Strategy"

    mod = importlib.import_module(mod_name)
    if not hasattr(mod, obj_name):
        raise AttributeError(f"Symbol '{obj_name}' not found in '{mod_name}'")
    return getattr(mod, obj_name)


def _canonical_json(obj: Any) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _hash_config(obj: Any, prefix: str = "cfg") -> str:
    s = _canonical_json(obj).encode("utf-8")
    h = hashlib.sha1(s).hexdigest()
    return f"{prefix}_{h[:12]}"


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _require(obj: Dict[str, Any], key: str, line_no: int) -> Any:
    if key not in obj:
        raise ConfigError(f"Line {line_no}: missing required key '{key}'")
    return obj[key]


def _as_float(x: Any, line_no: int, key: str) -> float:
    try:
        v = float(x)
    except Exception as e:
        raise ConfigError(f"Line {line_no}: '{key}' must be float-like") from e
    if not math.isfinite(v):
        raise ConfigError(f"Line {line_no}: '{key}' must be finite")
    return v


def _as_int(x: Any, line_no: int, key: str) -> int:
    try:
        v = int(x)
    except Exception as e:
        raise ConfigError(f"Line {line_no}: '{key}' must be int-like") from e
    return v


def _as_str(x: Any, line_no: int, key: str) -> str:
    if not isinstance(x, str):
        raise ConfigError(f"Line {line_no}: '{key}' must be a string")
    s = x.strip()
    if not s:
        raise ConfigError(f"Line {line_no}: '{key}' must be non-empty")
    return s


def _as_opt_str(x: Any, line_no: int, key: str) -> Optional[str]:
    if x is None:
        return None
    return _as_str(x, line_no, key)


def _as_list(x: Any, line_no: int, key: str) -> List[Any]:
    if x is None:
        return []
    if not isinstance(x, list):
        raise ConfigError(f"Line {line_no}: '{key}' must be a list")
    return x


# ============================================================
# Config parsing / validation
# ============================================================

# Option 2 bounds (DNF-lite)
MAX_ENTRY_ANY_CLAUSES = 3
MAX_ENTRY_ANY_CONDS_PER_CLAUSE = 6

def _parse_guardrails(
    obj: Optional[Dict[str, Any]],
    line_no: int,
) -> Optional[GuardrailsSpec]:
    if obj is None:
        return None
    if not isinstance(obj, dict):
        raise ConfigError(f"Line {line_no}: 'guardrails' must be an object")

    def opt_float(k: str) -> Optional[float]:
        if k not in obj or obj[k] is None:
            return None
        return _as_float(obj[k], line_no, f"guardrails.{k}")

    def opt_int(k: str, default: int) -> int:
        if k not in obj or obj[k] is None:
            return int(default)
        return _as_int(obj[k], line_no, f"guardrails.{k}")

    def opt_bool(k: str, default: bool) -> bool:
        if k not in obj or obj[k] is None:
            return bool(default)
        if not isinstance(obj[k], bool):
            raise ConfigError(f"Line {line_no}: guardrails.{k} must be bool")
        return bool(obj[k])

    return GuardrailsSpec(
        min_equity_stop=opt_float("min_equity_stop"),
        max_daily_loss_pct=opt_float("max_daily_loss_pct"),
        close_on_daily_limit=opt_bool("close_on_daily_limit", False),
        base_cooldown_minutes=opt_int("base_cooldown_minutes", 0),
        loss_streak_start=opt_int("loss_streak_start", 0),
        cooldown_base_min=opt_int("cooldown_base_min", 15),
        cooldown_max_min=opt_int("cooldown_max_min", 24 * 60),
    )

def _parse_entry_any(
    xs: Any,
    line_no: int,
    key: str = "entry_any",
    *,
    max_clauses: int = MAX_ENTRY_ANY_CLAUSES,
    max_conds_per_clause: int = MAX_ENTRY_ANY_CONDS_PER_CLAUSE,
) -> List[List[EntryCondition]]:
    """
    Option 2 (bounded DNF-lite):
      - Outer list: OR clauses
      - Inner list: AND conditions within a clause
    """
    clauses = _as_list(xs, line_no, key)
    if len(clauses) == 0:
        raise ConfigError(f"Line {line_no}: '{key}' must be a non-empty list")

    if max_clauses > 0 and len(clauses) > int(max_clauses):
        raise ConfigError(
            f"Line {line_no}: '{key}' too many clauses "
            f"({len(clauses)} > {max_clauses})"
        )

    out: List[List[EntryCondition]] = []
    for ci, clause in enumerate(clauses):
        if not isinstance(clause, list):
            raise ConfigError(
                f"Line {line_no}: '{key}[{ci}]' must be a list of conditions"
            )
        if len(clause) == 0:
            raise ConfigError(
                f"Line {line_no}: '{key}[{ci}]' must be a non-empty list"
            )
        if max_conds_per_clause > 0 and len(clause) > int(max_conds_per_clause):
            raise ConfigError(
                f"Line {line_no}: '{key}[{ci}]' too many conditions "
                f"({len(clause)} > {max_conds_per_clause})"
            )
        out.append(_parse_entry_conditions(clause, line_no, f"{key}[{ci}]"))

    return out



def _parse_entry_conditions(
    xs: Any,
    line_no: int,
    key: str,
) -> List[EntryCondition]:
    out: List[EntryCondition] = []
    lst = _as_list(xs, line_no, key)
    if len(lst) == 0:
        raise ConfigError(f"Line {line_no}: '{key}' must be a non-empty list")

    allowed_ops = {">", "<", ">=", "<="}
    for i, obj in enumerate(lst):
        if not isinstance(obj, dict):
            raise ConfigError(f"Line {line_no}: '{key}[{i}]' must be object")

        indicator = _as_str(_require(obj, "indicator", line_no), line_no, "indicator")
        operator = _as_str(_require(obj, "operator", line_no), line_no, "operator")
        threshold = _as_float(
            _require(obj, "threshold", line_no),
            line_no,
            "threshold",
        )
        ref_indicator = _as_opt_str(
            obj.get("ref_indicator"),
            line_no,
            "ref_indicator",
        )

        if operator not in allowed_ops:
            raise ConfigError(
                f"Line {line_no}: '{key}[{i}].operator' invalid: {operator}"
            )

        out.append(
            EntryCondition(
                indicator=indicator,
                operator=operator,
                threshold=float(threshold),
                ref_indicator=ref_indicator,
            )
        )
    return out


def _parse_filter_conditions(
    xs: Any,
    line_no: int,
    key: str,
) -> List[FilterCondition]:
    out: List[FilterCondition] = []
    lst = _as_list(xs, line_no, key)

    allowed_ops = {">", "<", ">=", "<="}
    for i, obj in enumerate(lst):
        if not isinstance(obj, dict):
            raise ConfigError(f"Line {line_no}: '{key}[{i}]' must be object")

        indicator = _as_str(_require(obj, "indicator", line_no), line_no, "indicator")
        operator = _as_str(_require(obj, "operator", line_no), line_no, "operator")
        threshold = _as_float(
            _require(obj, "threshold", line_no),
            line_no,
            "threshold",
        )
        ref_indicator = _as_opt_str(
            obj.get("ref_indicator"),
            line_no,
            "ref_indicator",
        )

        if operator not in allowed_ops:
            raise ConfigError(
                f"Line {line_no}: '{key}[{i}].operator' invalid: {operator}"
            )

        out.append(
            FilterCondition(
                indicator=indicator,
                operator=operator,
                threshold=float(threshold),
                ref_indicator=ref_indicator,
            )
        )
    return out


def _parse_risk(obj: Any, line_no: int) -> RiskConfig:
    if not isinstance(obj, dict):
        raise ConfigError(f"Line {line_no}: 'risk' must be an object")

    risk_per_trade_pct = _as_float(
        _require(obj, "risk_per_trade_pct", line_no),
        line_no,
        "risk.risk_per_trade_pct",
    )
    max_leverage = _as_float(
        _require(obj, "max_leverage", line_no),
        line_no,
        "risk.max_leverage",
    )
    sl_type = _as_str(_require(obj, "sl_type", line_no), line_no, "risk.sl_type")
    sl_param = _as_float(_require(obj, "sl_param", line_no), line_no, "risk.sl_param")

    sizing_mode = obj.get("sizing_mode", "legacy")
    if not isinstance(sizing_mode, str):
        raise ConfigError(f"Line {line_no}: risk.sizing_mode must be str")
    sizing_mode = sizing_mode.strip().lower()
    if sizing_mode not in {"legacy", "dynamic_leverage"}:
        raise ConfigError(
            f"Line {line_no}: risk.sizing_mode must be 'legacy' or 'dynamic_leverage'"
        )

    def opt_float(k: str, default: float) -> float:
        if k not in obj or obj[k] is None:
            return float(default)
        return _as_float(obj[k], line_no, f"risk.{k}")

    def opt_str(k: str, default: str) -> str:
        if k not in obj or obj[k] is None:
            return str(default)
        return _as_str(obj[k], line_no, f"risk.{k}")

    leverage_min = opt_float("leverage_min", 2.0)
    leverage_max = opt_float("leverage_max", 15.0)
    risk_max_pct = opt_float("risk_max_pct", 0.05)
    stop_floor_pct = opt_float("stop_floor_pct", 0.003)
    liq_safety_frac = opt_float("liq_safety_frac", 0.20)
    cost_bps_assumption = opt_float("cost_bps_assumption", 12.0)

    conf_model = opt_str("conf_model", "compression").strip().lower()
    if conf_model not in {"compression", "expansion"}:
        raise ConfigError(
            f"Line {line_no}: risk.conf_model must be 'compression' or 'expansion'"
        )

    conf_adx_lo = opt_float("conf_adx_lo", 15.0)
    conf_adx_hi = opt_float("conf_adx_hi", 30.0)
    conf_bbw_lo = opt_float("conf_bbw_lo", 0.04)
    conf_bbw_hi = opt_float("conf_bbw_hi", 0.08)
    conf_rvol_lo = opt_float("conf_rvol_lo", 1.0)
    conf_rvol_hi = opt_float("conf_rvol_hi", 2.0)
    conf_w_adx = opt_float("conf_w_adx", 0.5)
    conf_w_bbw = opt_float("conf_w_bbw", 0.3)
    conf_w_rvol = opt_float("conf_w_rvol", 0.2)

    if leverage_min <= 0 or leverage_max <= 0 or leverage_min > leverage_max:
        raise ConfigError(f"Line {line_no}: invalid leverage_min/leverage_max")
    if risk_max_pct <= 0 or risk_max_pct >= 1:
        raise ConfigError(f"Line {line_no}: risk_max_pct must be in (0,1)")
    if stop_floor_pct <= 0 or stop_floor_pct >= 0.20:
        raise ConfigError(f"Line {line_no}: stop_floor_pct must be in (0,0.2)")
    if liq_safety_frac < 0 or liq_safety_frac >= 0.95:
        raise ConfigError(f"Line {line_no}: liq_safety_frac must be in [0,0.95)")
    if cost_bps_assumption < 0 or cost_bps_assumption > 200:
        raise ConfigError(f"Line {line_no}: cost_bps_assumption seems too large")

    tp_is_market = obj.get("tp_is_market", None)
    if tp_is_market is None:
        tp_is_market = False
    if not isinstance(tp_is_market, bool):
        raise ConfigError(f"Line {line_no}: risk.tp_is_market must be bool")

    tp_r_multiples = [
        _as_float(v, line_no, "risk.tp_r_multiples[]")
        for v in _as_list(obj.get("tp_r_multiples"), line_no, "risk.tp_r_multiples")
    ]
    tp_fractions = [
        _as_float(v, line_no, "risk.tp_fractions[]")
        for v in _as_list(obj.get("tp_fractions"), line_no, "risk.tp_fractions")
    ]

    move_to_be_at_r = obj.get("move_to_be_at_r", None)
    if move_to_be_at_r is not None:
        move_to_be_at_r = _as_float(
            move_to_be_at_r,
            line_no,
            "risk.move_to_be_at_r",
        )

    if risk_per_trade_pct <= 0 or risk_per_trade_pct >= 1:
        raise ConfigError(f"Line {line_no}: risk.risk_per_trade_pct must be in (0, 1)")
    if max_leverage <= 0:
        raise ConfigError(f"Line {line_no}: risk.max_leverage must be > 0")

    if sl_type not in {"ATR", "PCT"}:
        raise ConfigError(f"Line {line_no}: risk.sl_type must be 'ATR' or 'PCT'")
    if sl_param <= 0:
        raise ConfigError(f"Line {line_no}: risk.sl_param must be > 0")

    if len(tp_r_multiples) != len(tp_fractions):
        raise ConfigError(
            f"Line {line_no}: risk.tp_r_multiples and risk.tp_fractions must match"
        )

    # Engine semantics: fractions are "fraction of initial" per level and final
    # TP closes remainder anyway. So no sum-to-1 requirement.
    for i, frac in enumerate(tp_fractions):
        if frac <= 0 or frac > 1.0:
            raise ConfigError(f"Line {line_no}: risk.tp_fractions[{i}] must be in (0, 1]")

    return RiskConfig(
        risk_per_trade_pct=float(risk_per_trade_pct),
        max_leverage=float(max_leverage),
        sl_type=str(sl_type),
        sl_param=float(sl_param),
        tp_is_market=bool(tp_is_market),
        sizing_mode=str(sizing_mode),
        leverage_min=float(leverage_min),
        leverage_max=float(leverage_max),
        risk_max_pct=float(risk_max_pct),
        stop_floor_pct=float(stop_floor_pct),
        liq_safety_frac=float(liq_safety_frac),
        cost_bps_assumption=float(cost_bps_assumption),
        conf_model=str(conf_model),
        conf_adx_lo=float(conf_adx_lo),
        conf_adx_hi=float(conf_adx_hi),
        conf_bbw_lo=float(conf_bbw_lo),
        conf_bbw_hi=float(conf_bbw_hi),
        conf_rvol_lo=float(conf_rvol_lo),
        conf_rvol_hi=float(conf_rvol_hi),
        conf_w_adx=float(conf_w_adx),
        conf_w_bbw=float(conf_w_bbw),
        conf_w_rvol=float(conf_w_rvol),
        tp_r_multiples=[float(x) for x in tp_r_multiples],
        tp_fractions=[float(x) for x in tp_fractions],
        move_to_be_at_r=float(move_to_be_at_r)
        if move_to_be_at_r is not None
        else None,
    )


def parse_strategy_config(
    obj: Dict[str, Any],
    line_no: int,
) -> Tuple[str, StrategyConfig, Dict[str, Any]]:
    """
    Returns (config_id, StrategyConfig, normalized_dict)
    where config_id is a stable hash of the normalized dict.
    """
    if not isinstance(obj, dict):
        raise ConfigError(f"Line {line_no}: config must be an object")

    # Strategy name first (needed to decide parsing mode)
    strategy_name = obj.get("strategy_name", None)
    if strategy_name is not None:
        strategy_name = _as_str(strategy_name, line_no, "strategy_name")
    else:
        strategy_name = "universal"

    params = obj.get("params", None)
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ConfigError(f"Line {line_no}: 'params' must be an object if provided")

    side = _as_str(_require(obj, "side", line_no), line_no, "side").lower()
    if side not in {"long", "short"}:
        raise ConfigError(f"Line {line_no}: side must be 'long' or 'short'")

    is_dca = str(strategy_name).strip().lower() in {"dca", "dca_swing"}

    # Entry logic (backward compatible):
    # - Legacy: "entry_conditions": [...] (AND)
    # - Option 2: "entry_any": [[...], [...]] (OR-of-AND)
    entry_any: Optional[List[List[EntryCondition]]] = None
    entry_conditions: List[EntryCondition] = []

    if not is_dca:
        if "entry_any" in obj and obj.get("entry_any") is not None:
            entry_any = _parse_entry_any(obj.get("entry_any"), line_no, "entry_any")
            entry_conditions = []
        else:
            entry_conditions = _parse_entry_conditions(
                _require(obj, "entry_conditions", line_no),
                line_no,
                "entry_conditions",
            )

    filters = _parse_filter_conditions(obj.get("filters", []), line_no, "filters")

    # DCA strategies don't need risk/entry parsing, but StrategyConfig requires RiskConfig.
    if is_dca:
        risk = RiskConfig()
    else:
        risk = _parse_risk(_require(obj, "risk", line_no), line_no)

    guardrails = _parse_guardrails(obj.get("guardrails", None), line_no)

    normalized = {
        "strategy_name": strategy_name,
        "side": side,
        "entry_conditions": [asdict(x) for x in entry_conditions],
        "entry_any": (
            [[asdict(c) for c in clause] for clause in entry_any]
            if entry_any is not None
            else None
        ),
        "filters": [asdict(x) for x in filters],
        "risk": asdict(risk),
        "guardrails": asdict(guardrails) if guardrails is not None else None,
        "params": params,
    }
    cfg_id = _hash_config(normalized)

    cfg = StrategyConfig(
        strategy_name=strategy_name,
        side=side,
        entry_conditions=entry_conditions,
        entry_any=entry_any,
        filters=filters,
        risk=risk,
        guardrails=guardrails,
        params=params,
    )
    return cfg_id, cfg, normalized


# ============================================================
# Gates / detectors
# ============================================================


def _best_trade_dominance(trades_df: pd.DataFrame) -> Dict[str, Any]:
    if trades_df is None or trades_df.empty or "net_pnl" not in trades_df.columns:
        return {
            "wins_sum": 0.0,
            "net_sum": 0.0,
            "best_trade": 0.0,
            "best_over_wins_sum": 0.0,
            "best_over_abs_net_sum": 0.0,
        }

    pnl = pd.to_numeric(trades_df["net_pnl"], errors="coerce").fillna(0.0).astype(float)
    wins_sum = float(pnl[pnl > 0].sum())
    net_sum = float(pnl.sum())
    best = float(pnl.max()) if len(pnl) else 0.0

    best_over_wins = (best / wins_sum) if wins_sum > 1e-12 else 0.0
    best_over_abs_net = (best / abs(net_sum)) if abs(net_sum) > 1e-12 else 0.0

    return {
        "wins_sum": wins_sum,
        "net_sum": net_sum,
        "best_trade": best,
        "best_over_wins_sum": float(best_over_wins),
        "best_over_abs_net_sum": float(best_over_abs_net),
    }

def _complexity_stats(cfg: StrategyConfig) -> Dict[str, Any]:
    """
    Cheap, config-only complexity stats (helps robustness + overfit control).
    """
    entry_any = getattr(cfg, "entry_any", None)
    if entry_any:
        entry_clauses = int(len(entry_any))
        entry_conds = int(sum(len(clause) for clause in entry_any))
        logic = "entry_any"
        atoms = [(c.indicator, c.ref_indicator) for clause in entry_any for c in clause]
        inds = [c.indicator for clause in entry_any for c in clause]
    else:
        entry_clauses = 1 if len(cfg.entry_conditions) else 0
        entry_conds = int(len(cfg.entry_conditions))
        logic = "entry_conditions"
        atoms = [(c.indicator, c.ref_indicator) for c in cfg.entry_conditions]
        inds = [c.indicator for c in cfg.entry_conditions]

    filter_conds = int(len(cfg.filters))
    total_conds = int(entry_conds + filter_conds)

    # Include filters in uniqueness counts (filters are real logic too)
    atoms.extend([(f.indicator, f.ref_indicator) for f in cfg.filters])
    inds.extend([f.indicator for f in cfg.filters])

    # unique atoms: indicator + ref_indicator (better proxy for distinct checks)
    uniq_atoms = set((a, b) for (a, b) in atoms)
    uniq_inds = set(inds)

    return {
        "logic": logic,
        "entry_clauses": entry_clauses,
        "entry_conds": entry_conds,
        "filter_conds": filter_conds,
        "total_conds": total_conds,
        "unique_atoms": int(len(uniq_atoms)),
        "unique_indicators": int(len(uniq_inds)),
    }

def _passes_gates(
    row: Dict[str, Any],
    min_trades: Optional[int],
    min_sqn: Optional[float],
    max_fee_impact_pct: Optional[float],
    max_best_over_wins: Optional[float],
    max_entry_clauses: Optional[int] = None,
    max_total_conds: Optional[int] = None,
    max_unique_atoms: Optional[int] = None,
) -> Tuple[bool, str]:
    trades = int(row.get("trades_summary.trades_closed", 0) or 0)
    sqn = _safe_float(row.get("trades_summary.sqn", 0.0))
    fee_impact = _safe_float(row.get("efficiency.fee_impact_pct", 0.0))
    best_over_wins = _safe_float(row.get("dominance.best_over_wins_sum", 0.0))

    entry_clauses = int(row.get("complexity.entry_clauses", 0) or 0)
    total_conds = int(row.get("complexity.total_conds", 0) or 0)
    unique_atoms = int(row.get("complexity.unique_atoms", 0) or 0)

    if min_trades is not None and trades < int(min_trades):
        return False, f"min_trades({trades}<{min_trades})"
    if min_sqn is not None and sqn < float(min_sqn):
        return False, f"min_sqn({sqn:.2f}<{min_sqn})"
    if max_fee_impact_pct is not None and fee_impact > float(max_fee_impact_pct):
        return False, f"fee_impact({fee_impact:.1f}>{max_fee_impact_pct})"
    if max_best_over_wins is not None and best_over_wins > float(max_best_over_wins):
        return False, f"best_trade_dominance({best_over_wins:.2f}>{max_best_over_wins})"

    if max_entry_clauses is not None and entry_clauses > int(max_entry_clauses):
        return False, f"max_entry_clauses({entry_clauses}>{max_entry_clauses})"
    if max_total_conds is not None and total_conds > int(max_total_conds):
        return False, f"max_total_conds({total_conds}>{max_total_conds})"
    if max_unique_atoms is not None and unique_atoms > int(max_unique_atoms):
        return False, f"max_unique_atoms({unique_atoms}>{max_unique_atoms})"

    return True, ""


# ============================================================
# Data handling
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


def _make_constraints():
    from engine.contracts import EngineConstraints

    return EngineConstraints(
        max_leverage=20.0,
        maint_margin_rate=0.005,
        price_tick=0.1,
        qty_step=0.001,
        min_notional_usdt=5.0,
    )


def _read_jsonl(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config grid not found: {p}")

    # Windows PowerShell often writes UTF-8 with BOM. json.loads() rejects BOM,
    # so read with utf-8-sig to transparently strip it if present.
    with open(p, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            try:
                obj = json.loads(s)
            except Exception as e:
                raise ConfigError(f"Line {line_no}: invalid JSON") from e
            if not isinstance(obj, dict):
                raise ConfigError(f"Line {line_no}: each JSONL row must be an object")
            yield line_no, obj


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(_canonical_json(r))
            f.write("\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        pd.DataFrame([]).to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)


# ============================================================
# Backtest runners (sweep  full rerun)
# ============================================================


def _supports_fast_flags() -> Dict[str, bool]:
    sig = inspect.signature(run_backtest_once)
    return {
        "record_equity_curve": "record_equity_curve" in sig.parameters,
        "record_fills": "record_fills" in sig.parameters,
    }


def _instantiate_template(template_cls: Any, cfg: StrategyConfig) -> Any:
    try:
        return template_cls(config=cfg)
    except TypeError:
        return template_cls(cfg)

def _chunked(xs: List[Any], n: int) -> List[List[Any]]:
    n = int(max(1, n))
    return [xs[i : i + n] for i in range(0, len(xs), n)]


def _worker_init(df_feat_path: str, template_dotted: str, market_mode: str) -> None:
    """
    Worker initializer (runs once per process).
    Loads df_feat and imports the strategy template class.
    """
    global _WORKER_DF_FEAT, _WORKER_TEMPLATE_CLS, _WORKER_CONSTRAINTS, _WORKER_ENGINE_CFG

    _WORKER_DF_FEAT = pd.read_parquet(df_feat_path)
    _WORKER_TEMPLATE_CLS = _import_symbol(template_dotted)
    _WORKER_CONSTRAINTS = _make_constraints()
    _WORKER_ENGINE_CFG = BacktestConfig(market_mode=str(market_mode).lower())


def _eval_sweep_row(
    *,
    cfg_id: str,
    line_no: int,
    cfg: StrategyConfig,
    seed: int,
    starting_equity: float,
    fast_sweep: bool,
) -> Dict[str, Any]:
    """
    Evaluate one config and return a flattened sweep row (without gates applied).

    This is used in:
      - single-process sweep (we set globals in-process)
      - multiprocessing sweep (globals are set by _worker_init)
    """
    df_feat = _WORKER_DF_FEAT
    template_cls = _WORKER_TEMPLATE_CLS
    constraints = _WORKER_CONSTRAINTS
    engine_cfg = _WORKER_ENGINE_CFG
    if df_feat is None or template_cls is None or constraints is None or engine_cfg is None:
        raise RuntimeError("Worker context not initialized (df/template/constraints/cfg).")

    is_spot = str(getattr(engine_cfg, "market_mode", "perps")).lower() == "spot"

    # Spot/DCA: we must record equity curve to compute cashflow-aware performance (TWR).
    rec_eq = True if is_spot else (False if fast_sweep else True)
    rec_fills = False if fast_sweep else True

    strategy = _instantiate_template(template_cls, cfg)
    t1 = time.time()
    metrics, fills_df, eq_df, trades_df, _guard = _run_once(
        df_feat=df_feat,
        strategy=strategy,
        seed=int(seed),
        starting_equity=float(starting_equity),
        constraints=constraints,
        engine_cfg=engine_cfg,
        show_progress=False,
        features_ready=True,
        record_equity_curve=rec_eq,
        record_fills=rec_fills,
    )
    elapsed = time.time() - t1

    # Cheap stats: always available from trades_df + metrics
    tstats = _trade_stats(trades_df)
    efficiency = _efficiency_stats(trades_df)
    dominance = _best_trade_dominance(trades_df)

    # These may be empty in fast sweep (no eq_df/fills_df)
    if is_spot:
        perf = _build_cashflow_performance_stats(df_feat, eq_df)
    else:
        perf = _build_performance_stats(df_feat, eq_df)
    exits = _exit_counts(trades_df)
    tp_levels = _tp_level_counts(trades_df)
    stop_moves = _stop_move_stats(trades_df)
    exec_break = _execution_breakdown(fills_df)
    exposure = _exposure_stats(eq_df, fills_df)

    fees_total = (
        float(eq_df["fees_paid_total"].iloc[-1])
        if (eq_df is not None and len(eq_df) and "fees_paid_total" in eq_df.columns)
        else 0.0
    )
    funding_net = (
        float(eq_df["funding_net"].iloc[-1])
        if (eq_df is not None and len(eq_df) and "funding_net" in eq_df.columns)
        else 0.0
    )
    # Recon diff (might be 0/NaN if eq_df not recorded)
    recon_diff = 0.0
    try:
        if (
            eq_df is not None
            and not eq_df.empty
            and trades_df is not None
            and not trades_df.empty
            and "equity" in eq_df.columns
        ):
            eq_delta = float(eq_df["equity"].iloc[-1]) - float(eq_df["equity"].iloc[0])
            trade_sum = float(
                pd.to_numeric(trades_df["net_pnl"], errors="coerce").fillna(0.0).sum()
            )
            cashflow_total = 0.0
            if "cashflow" in eq_df.columns:
                cashflow_total = float(
                    pd.to_numeric(eq_df["cashflow"], errors="coerce")
                    .fillna(0.0)
                    .sum()
                )
            if is_spot:
                recon_diff = float(eq_delta - cashflow_total - trade_sum)
            else:
                recon_diff = float(eq_delta - trade_sum)
    except Exception:
        recon_diff = float("nan")

    full = {
        "config": {
            "id": str(cfg_id),
            "line_no": int(line_no),
            "strategy_name": cfg.strategy_name,
            "side": cfg.side,
            "label": _config_label(cfg),
            "params": dict(getattr(cfg, "params", {}) or {}),
        },
        "complexity": _complexity_stats(cfg),
        "runtime": {"elapsed_sec": float(elapsed)},
        "equity": metrics.get("equity", {}),
        "performance": perf,
        "trades_summary": tstats,
        "exit_counts": exits,
        "tp_level_counts": tp_levels,
        "stop_move_stats": stop_moves,
        "efficiency": efficiency,
        "execution_breakdown": exec_break,
        "exposure": exposure,
        "dominance": dominance,
        "costs": {
            "fees_paid_total": float(fees_total),
            "funding_net": float(funding_net),
        },
        "reconciliation": {"diff": float(recon_diff)},
    }

    return _flatten(full)

def _worker_run_sweep_chunk(
    chunk: List[Tuple[str, int, StrategyConfig]],
    seed: int,
    starting_equity: float,
    fast_sweep: bool,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cfg_id, line_no, cfg in chunk:
        out.append(
            _eval_sweep_row(
                cfg_id=str(cfg_id),
                line_no=int(line_no),
                cfg=cfg,
                seed=int(seed),
                starting_equity=float(starting_equity),
                fast_sweep=bool(fast_sweep),
            )
        )
    return out

def _worker_run_rerun_chunk(
    chunk: List[Tuple[str, int, StrategyConfig]],
    seed: int,
    starting_equity: float,
) -> List[Dict[str, Any]]:
    """
    Full rerun (high fidelity) chunk runner for multiprocessing.
    Uses the same evaluation function but forces fast_sweep=False (records eq/fills).
    """
    out: List[Dict[str, Any]] = []
    for cfg_id, line_no, cfg in chunk:
        out.append(
            _eval_sweep_row(
                cfg_id=str(cfg_id),
                line_no=int(line_no),
                cfg=cfg,
                seed=int(seed),
                starting_equity=float(starting_equity),
                fast_sweep=False,
            )
        )
    return out

def _run_once(
    *,
    df_feat: pd.DataFrame,
    strategy: Any,
    seed: int,
    starting_equity: float,
    constraints: Any,
    engine_cfg: BacktestConfig,
    features_ready: bool,
    show_progress: bool,
    record_equity_curve: Optional[bool] = None,
    record_fills: Optional[bool] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Calls run_backtest_once with optional fast-mode flags if supported by the engine.
    Compatible with older engine signatures (will ignore flags).
    """
    kwargs: Dict[str, Any] = {
        "df": df_feat,
        "strategy": strategy,
        "seed": int(seed),
        "starting_equity": float(starting_equity),
        "constraints": constraints,
        "cfg": engine_cfg,
        "show_progress": bool(show_progress),
        "features_ready": bool(features_ready),
    }

    if record_equity_curve is not None:
        kwargs["record_equity_curve"] = bool(record_equity_curve)
    if record_fills is not None:
        kwargs["record_fills"] = bool(record_fills)

    try:
        return run_backtest_once(**kwargs)
    except TypeError:
        # Engine doesn't support some kwargs (fast flags). Retry minimal call.
        kwargs.pop("record_equity_curve", None)
        kwargs.pop("record_fills", None)
        return run_backtest_once(**kwargs)


def _save_artifacts_for_config(
    out_dir: Path,
    cfg_obj: Dict[str, Any],
    df_feat: pd.DataFrame,
    template_cls: Any,
    seed: int,
    starting_equity: float,
    market_mode: str,
) -> None:
    constraints = _make_constraints()
    engine_cfg = BacktestConfig(market_mode=str(market_mode).lower())

    strategy = _instantiate_template(template_cls, cfg_obj["__parsed__"])

    metrics, fills_df, eq_df, trades_df, _guard = _run_once(
        df_feat=df_feat,
        strategy=strategy,
        seed=seed,
        starting_equity=starting_equity,
        constraints=constraints,
        engine_cfg=engine_cfg,
        features_ready=True,
        show_progress=False,
        record_equity_curve=True,
        record_fills=True,
    )

    perf = _build_performance_stats(df_feat, eq_df)
    tstats = _trade_stats(trades_df)
    exits = _exit_counts(trades_df)
    tp_levels = _tp_level_counts(trades_df)
    stop_moves = _stop_move_stats(trades_df)
    efficiency = _efficiency_stats(trades_df)
    exec_break = _execution_breakdown(fills_df)
    exposure = _exposure_stats(eq_df, fills_df)
    dominance = _best_trade_dominance(trades_df)

    fees_total = (
        float(eq_df["fees_paid_total"].iloc[-1])
        if (eq_df is not None and len(eq_df) and "fees_paid_total" in eq_df.columns)
        else 0.0
    )
    funding_net = (
        float(eq_df["funding_net"].iloc[-1])
        if (eq_df is not None and len(eq_df) and "funding_net" in eq_df.columns)
        else 0.0
    )

    metrics["performance"] = perf
    metrics["trades_summary"] = tstats
    metrics["exit_counts"] = exits
    metrics["tp_level_counts"] = tp_levels
    metrics["stop_move_stats"] = stop_moves
    metrics["efficiency"] = efficiency
    metrics["execution_breakdown"] = exec_break
    metrics["exposure"] = exposure
    metrics["dominance"] = dominance
    metrics["costs"] = {"fees_paid_total": fees_total, "funding_net": funding_net}
    metrics["tp_sl_profitability"] = _tp_sl_profitability_report(trades_df)

    # Recon
    try:
        if (
            eq_df is not None
            and not eq_df.empty
            and trades_df is not None
            and not trades_df.empty
            and "equity" in eq_df.columns
        ):
            eq_delta = float(eq_df["equity"].iloc[-1]) - float(eq_df["equity"].iloc[0])
            trade_sum = float(
                pd.to_numeric(trades_df["net_pnl"], errors="coerce").fillna(0.0).sum()
            )
            metrics["reconciliation"] = {
                "equity_delta": float(eq_delta),
                "sum_trade_net_pnl": float(trade_sum),
                "diff": float(eq_delta - trade_sum),
            }
    except Exception:
        metrics["reconciliation"] = {"error": "failed_to_compute"}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(cfg_obj["normalized"], indent=2),
        encoding="utf-8",
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    if fills_df is not None and not fills_df.empty:
        fills_df.to_csv(out_dir / "fills.csv", index=False)
    if trades_df is not None and not trades_df.empty:
        trades_df.to_csv(out_dir / "trades.csv", index=False)
    if eq_df is not None and not eq_df.empty:
        eq_df.to_csv(out_dir / "equity_curve.csv", index=False)


# ============================================================
# CLI
# ============================================================


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Batch strategy runner (JSONL grid)")
    ap.add_argument("--data", required=True, help="Path to OHLCV CSV/Parquet")
    ap.add_argument("--grid", required=True, help="Path to JSONL configs grid")
    ap.add_argument(
        "--template",
        default="strategies.universal:UniversalStrategy",
        help="Strategy template class (must accept config=StrategyConfig)",
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--starting-equity", type=float, default=1000.0)
    ap.add_argument("--vol-window", type=int, default=60)
    ap.add_argument("--out", default="runs", help="Output root folder")
    ap.add_argument(
        "--run-name",
        default=None,
        help="Override run folder name (default: batch_YYYYmmdd_HHMMSS).",
    )

    ap.add_argument(
        "--market-mode",
        default="spot",
        choices=["spot", "perps"],
        help="Engine market mode (spot for DCA lab, perps for perp strategies).",
    )

    ap.add_argument(
        "--features-ready",
        action="store_true",
        help=(
            "Treat input --data as already feature-enriched (skip add_features). "
            "Use this when passing a prebuilt df_feat parquet."
        ),
    )

    ap.add_argument(
        "--rerun-jobs",
        type=int,
        default=0,
        help=(
            "Parallel workers for full rerun stage. "
            "0 = use --jobs. 1 = single-process rerun."
        ),
    )
    ap.add_argument(
        "--artifact-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress while saving top-k artifacts (can be slow).",
    )

    # Gates
    ap.add_argument("--min-trades", type=int, default=50)
    ap.add_argument("--min-sqn", type=float, default=None)
    ap.add_argument("--max-fee-impact-pct", type=float, default=150.0)
    ap.add_argument(
        "--max-best-over-wins",
        type=float,
        default=0.9,
        help="Best trade / total wins sum must be <= this",
    )

    # Complexity gates (optional; defaults are permissive)
    ap.add_argument(
        "--max-entry-clauses",
        type=int,
        default=None,
        help="Reject configs with entry clauses > this (Option 2).",
    )
    ap.add_argument(
        "--max-total-conds",
        type=int,
        default=None,
        help="Reject configs with (entry_conds + filter_conds) > this.",
    )
    ap.add_argument(
        "--max-unique-atoms",
        type=int,
        default=None,
        help="Reject configs with unique (indicator,ref) atoms > this.",
    )

    # Sorting  output
    ap.add_argument(
        "--sort-by",
        default="performance.sharpe",
        help="Final metric key to sort by (flattened, full rerun)",
    )
    ap.add_argument(
        "--sort-desc",
        action="store_true",
        help="Final sort descending (default ascending if not set)",
    )

    # Fast sweep  rerun
    ap.add_argument(
        "--fast-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a fast sweep (skip heavy recording) then rerun candidates fully",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel sweep workers (Windows spawn-safe). Default: 1",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help=(
            "Configs per worker task during sweep. 0 = auto. "
            "Windows: chunking matters a lot."
        ),
    )
    ap.add_argument(
        "--sweep-sort-by",
        default="equity.total_return",
        help="Sweep ranking key (flattened; must exist in sweep rows)",
    )
    ap.add_argument(
        "--sweep-sort-desc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sweep sort descending",
    )
    ap.add_argument(
        "--rerun-n",
        type=int,
        default=200,
        help="How many passing configs to rerun fully (buffer before top-k)",
    )

    ap.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Save full artifacts for top K passing configs (final ranking)",
    )
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args(argv)

    run_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_name = (
        _slug(str(args.run_name), max_len=180)
        if args.run_name
        else f"batch_{run_tag}"
    )
    out_dir = (Path(args.out) / str(run_name)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load  enrich once
    df = _load_ohlcv(args.data)
    df = _normalize_columns(df)
    df = _ensure_vol_bps(df, window=args.vol_window)
    if "liq_mult" not in df.columns:
        df["liq_mult"] = 1.0
    else:
        df["liq_mult"] = pd.to_numeric(df["liq_mult"], errors="coerce").fillna(1.0)

    t0 = time.time()
    if args.features_ready:
        df_feat = df
    else:
        df_feat = add_features(df)
    feat_sec = time.time() - t0

    # Resolve jobs
    jobs = int(max(1, args.jobs))
    rerun_jobs = int(args.rerun_jobs) if int(args.rerun_jobs) > 0 else jobs
    rerun_jobs = int(max(1, rerun_jobs))

    template_cls = _import_symbol(args.template)

    # Parse configs
    parsed_configs: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for line_no, raw in _read_jsonl(args.grid):
        try:
            cfg_id, cfg, normalized = parse_strategy_config(raw, line_no)
            parsed_configs.append(
                {
                    "config_id": cfg_id,
                    "line_no": line_no,
                    "config": raw,
                    "normalized": normalized,
                    "__parsed__": cfg,
                }
            )
        except Exception as e:
            errors.append({"line_no": line_no, "error": str(e), "raw": raw})

    if errors:
        _write_jsonl(out_dir / "errors.jsonl", errors)

    if len(parsed_configs) == 0:
        print(f"\nNo configs parsed successfully. Output: {out_dir}")
        if errors:
            print(f"Errors: {len(errors)}  See: {out_dir / 'errors.jsonl'}")
        return 1

    # Save resolved configs (normalized, stable IDs)
    resolved_rows: List[Dict[str, Any]] = []
    for x in parsed_configs:
        resolved_rows.append(
            {
                "config_id": x["config_id"],
                "line_no": x["line_no"],
                "normalized": x["normalized"],
            }
        )
    _write_jsonl(out_dir / "configs_resolved.jsonl", resolved_rows)

    constraints = _make_constraints()
    engine_cfg = BacktestConfig(market_mode=str(args.market_mode).lower())

    supports = _supports_fast_flags()
    if args.fast_sweep and (not supports["record_equity_curve"] or not supports["record_fills"]):
        # We can still run, but "fast sweep" won't actually be fast until engine adds flags
        # (backwards compatibility).
        print(
            "\nWarning: engine.run_backtest_once does not expose fast recording flags yet. "
            "Batch will run in full-record mode."
        )
    
    # If we use multiprocessing (Windows spawn), avoid sending df_feat to workers.
    # Save once and have workers load it in their initializer.
    df_feat_path = out_dir / "df_feat.parquet"
    if jobs > 1:
        t_save0 = time.time()
        df_feat.to_parquet(df_feat_path, index=False)
        t_save = time.time() - t_save0
        print(f"\nSaved df_feat for workers: {df_feat_path} ({t_save:.2f}s)")

    # Sweep loop
    rows_sweep: List[Dict[str, Any]] = []

    if jobs == 1:
        # Reuse the same evaluation logic as workers, but initialize globals in-process.
        global _WORKER_DF_FEAT, _WORKER_TEMPLATE_CLS, _WORKER_CONSTRAINTS, _WORKER_ENGINE_CFG
        _WORKER_DF_FEAT = df_feat
        _WORKER_TEMPLATE_CLS = template_cls
        _WORKER_CONSTRAINTS = constraints
        _WORKER_ENGINE_CFG = BacktestConfig(market_mode=str(args.market_mode).lower())

        it: Any = parsed_configs
        if not args.no_progress:
            try:
                from tqdm import tqdm

                it = tqdm(it, total=len(parsed_configs), desc="Batch", unit="cfg")
            except Exception:
                pass

        for x in it:
            row = _eval_sweep_row(
                cfg_id=str(x["config_id"]),
                line_no=int(x["line_no"]),
                cfg=x["__parsed__"],
                seed=int(args.seed),
                starting_equity=float(args.starting_equity),
                fast_sweep=bool(args.fast_sweep),
            )

            passed, reason = _passes_gates(
                row=row,
                min_trades=args.min_trades,
                min_sqn=args.min_sqn,
                max_fee_impact_pct=args.max_fee_impact_pct,
                max_best_over_wins=args.max_best_over_wins,
                max_entry_clauses=args.max_entry_clauses,
                max_total_conds=args.max_total_conds,
                max_unique_atoms=args.max_unique_atoms,
            )
            row["gates.passed"] = bool(passed)
            row["gates.reject_reason"] = str(reason)

            rows_sweep.append(row)
    else:
        # Parallel sweep (Windows spawn-safe): workers load df_feat from parquet.
        chunk_size = int(args.chunk_size) if int(args.chunk_size) > 0 else 20

        items: List[Tuple[str, int, StrategyConfig]] = [
            (str(x["config_id"]), int(x["line_no"]), x["__parsed__"]) for x in parsed_configs
        ]
        chunks = _chunked(items, chunk_size)

        use_tqdm = False
        pbar = None
        if not args.no_progress:
            try:
                from tqdm import tqdm

                pbar = tqdm(total=len(items), desc="Batch", unit="cfg")
                use_tqdm = True
            except Exception:
                use_tqdm = False

        with ProcessPoolExecutor(
            max_workers=jobs,
            initializer=_worker_init,
            initargs=(str(df_feat_path), str(args.template), str(args.market_mode)),
        ) as ex:
            futs = [
                ex.submit(
                    _worker_run_sweep_chunk,
                    chunk,
                    int(args.seed),
                    float(args.starting_equity),
                    bool(args.fast_sweep),
                )
                for chunk in chunks
            ]

            for fut in as_completed(futs):
                rows = fut.result()
                for row in rows:
                    passed, reason = _passes_gates(
                        row=row,
                        min_trades=args.min_trades,
                        min_sqn=args.min_sqn,
                        max_fee_impact_pct=args.max_fee_impact_pct,
                        max_best_over_wins=args.max_best_over_wins,
                        max_entry_clauses=args.max_entry_clauses,
                        max_total_conds=args.max_total_conds,
                        max_unique_atoms=args.max_unique_atoms,
                    )
                    row["gates.passed"] = bool(passed)
                    row["gates.reject_reason"] = str(reason)
                    rows_sweep.append(row)

                if use_tqdm and pbar is not None:
                    pbar.update(len(rows))

        if use_tqdm and pbar is not None:
            pbar.close()

    # Save sweep results (for all configs)
    _write_csv(out_dir / "results.csv", rows_sweep)

    df_sweep = pd.DataFrame(rows_sweep)
    df_pass_sweep = df_sweep[df_sweep["gates.passed"].astype(bool)].copy()
    df_pass_sweep.to_csv(out_dir / "results_passed.csv", index=False)

    # If nothing passed gates, stop gracefully (common for DCA if min-trades defaults are high).
    if df_pass_sweep.empty:
        _write_csv(out_dir / "results_full.csv", [])
        _write_csv(out_dir / "results_full_passed.csv", [])
        meta = {
            "data": args.data,
            "grid": args.grid,
            "template": args.template,
            "market_mode": str(args.market_mode),
            "seed": int(args.seed),
            "starting_equity": float(args.starting_equity),
            "configs_total": int(len(parsed_configs)),
            "errors_total": int(len(errors)),
            "note": "No configs passed gates. Try --min-trades 0 for DCA sweeps.",
            "out_dir": str(out_dir),
        }
        (out_dir / "batch_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"\nBatch complete (no passing configs). Output: {out_dir}")
        return 0

    # Choose which configs to rerun fully
    rerun_n = int(max(args.top_k, args.rerun_n))
    sweep_sort_key = args.sweep_sort_by
    if sweep_sort_key not in df_pass_sweep.columns:
        sweep_sort_key = "equity.total_return"

    df_candidates = df_pass_sweep.sort_values(
        sweep_sort_key,
        ascending=(not args.sweep_sort_desc),
    ).head(rerun_n)

    candidate_ids = df_candidates["config.id"].tolist()
    id_to_cfg = {x["config_id"]: x for x in parsed_configs}
    candidates = [id_to_cfg[cid] for cid in candidate_ids if cid in id_to_cfg]

    # Full rerun for candidates (for correct final sorting, plus artifacts)
    rows_full: List[Dict[str, Any]] = []
    if len(candidates) == 0:
        rows_full = []
    elif rerun_jobs == 1:
        it2: Any = candidates
        if not args.no_progress:
            try:
                from tqdm import tqdm

                it2 = tqdm(it2, total=len(candidates), desc="Rerun", unit="cfg")
            except Exception:
                pass

        for x in it2:
            row = _eval_sweep_row(
                cfg_id=str(x["config_id"]),
                line_no=int(x["line_no"]),
                cfg=x["__parsed__"],
                seed=int(args.seed),
                starting_equity=float(args.starting_equity),
                fast_sweep=False,
            )
            passed, reason = _passes_gates(
                row=row,
                min_trades=args.min_trades,
                min_sqn=args.min_sqn,
                max_fee_impact_pct=args.max_fee_impact_pct,
                max_best_over_wins=args.max_best_over_wins,
                max_entry_clauses=args.max_entry_clauses,
                max_total_conds=args.max_total_conds,
                max_unique_atoms=args.max_unique_atoms,
            )
            row["gates.passed"] = bool(passed)
            row["gates.reject_reason"] = str(reason)
            rows_full.append(row)
    else:
        # Parallel full rerun: ensure df_feat_path exists for workers
        if not df_feat_path.exists():
            df_feat.to_parquet(df_feat_path, index=False)
        chunk_size_r = int(args.chunk_size) if int(args.chunk_size) > 0 else 10
        items2: List[Tuple[str, int, StrategyConfig]] = [
            (str(x["config_id"]), int(x["line_no"]), x["__parsed__"]) for x in candidates
        ]
        chunks2 = _chunked(items2, chunk_size_r)

        use_tqdm = False
        pbar = None
        if not args.no_progress:
            try:
                from tqdm import tqdm

                pbar = tqdm(total=len(items2), desc="Rerun", unit="cfg")
                use_tqdm = True
            except Exception:
                use_tqdm = False

        with ProcessPoolExecutor(
            max_workers=rerun_jobs,
            initializer=_worker_init,
            initargs=(str(df_feat_path), str(args.template), str(args.market_mode)),
        ) as ex:
            futs2 = [
                ex.submit(
                    _worker_run_rerun_chunk,
                    ch,
                    int(args.seed),
                    float(args.starting_equity),
                )
                for ch in chunks2
            ]

            for fut in as_completed(futs2):
                rows = fut.result()
                for row in rows:
                    passed, reason = _passes_gates(
                        row=row,
                        min_trades=args.min_trades,
                        min_sqn=args.min_sqn,
                        max_fee_impact_pct=args.max_fee_impact_pct,
                        max_best_over_wins=args.max_best_over_wins,
                        max_entry_clauses=args.max_entry_clauses,
                        max_total_conds=args.max_total_conds,
                        max_unique_atoms=args.max_unique_atoms,
                    )
                    row["gates.passed"] = bool(passed)
                    row["gates.reject_reason"] = str(reason)
                    rows_full.append(row)

                if use_tqdm and pbar is not None:
                    pbar.update(len(rows))

        if use_tqdm and pbar is not None:
            pbar.close()

    _write_csv(out_dir / "results_full.csv", rows_full)
    df_full = pd.DataFrame(rows_full)
    df_full_pass = df_full[df_full["gates.passed"].astype(bool)].copy()
    if df_full.empty or "gates.passed" not in df_full.columns:
        df_full_pass = pd.DataFrame([])
    else:
        df_full_pass = df_full[df_full["gates.passed"].astype(bool)].copy()
    df_full_pass.to_csv(out_dir / "results_full_passed.csv", index=False)

    # Final ranking based on full metrics
    sort_key = args.sort_by
    if sort_key not in df_full_pass.columns:
        sort_key = "equity.total_return"

    df_rank = df_full_pass.sort_values(sort_key, ascending=(not args.sort_desc))
    k = int(max(0, args.top_k))

    if k > 0 and len(df_rank) > 0:
        top_dir = out_dir / "top"
        top_dir.mkdir(parents=True, exist_ok=True)

        top_ids = df_rank["config.id"].head(k).tolist()

        it3: Any = list(enumerate(top_ids, start=1))
        if args.artifact_progress and (not args.no_progress):
            try:
                from tqdm import tqdm

                it3 = tqdm(it3, total=len(top_ids), desc="Artifacts", unit="cfg")
            except Exception:
                pass

        for rank, cfg_id in it3:
            x = id_to_cfg.get(cfg_id)
            if x is None:
                continue
            label = _config_label(x["__parsed__"])
            cfg_folder = top_dir / f"{rank:04d}_{cfg_id}_{label}"
            _save_artifacts_for_config(
                out_dir=cfg_folder,
                cfg_obj=x,
                df_feat=df_feat,
                template_cls=template_cls,
                seed=int(args.seed),
                starting_equity=float(args.starting_equity),
                market_mode=str(args.market_mode),
            )

    meta = {
        "data": args.data,
        "grid": args.grid,
        "template": args.template,
        "seed": int(args.seed),
        "starting_equity": float(args.starting_equity),
        "vol_window": int(args.vol_window),
        "feature_compute_sec": float(feat_sec),
        "features_ready": bool(args.features_ready),
        "market_mode": str(args.market_mode),
        "configs_total": int(len(parsed_configs)),
        "errors_total": int(len(errors)),
        "fast_sweep": bool(args.fast_sweep),
        "engine_supports_fast_flags": supports,
        "sweep_sort_by": args.sweep_sort_by,
        "sweep_sort_desc": bool(args.sweep_sort_desc),
        "rerun_n": int(rerun_n),
        "final_sort_by": args.sort_by,
        "final_sort_desc": bool(args.sort_desc),
        "top_k": int(args.top_k),
        "gates": {
            "min_trades": int(args.min_trades) if args.min_trades is not None else None,
            "min_sqn": float(args.min_sqn) if args.min_sqn is not None else None,
            "max_fee_impact_pct": (
                float(args.max_fee_impact_pct)
                if args.max_fee_impact_pct is not None
                else None
            ),
            "max_best_over_wins": (
                float(args.max_best_over_wins)
                if args.max_best_over_wins is not None
                else None
            ),
            "max_entry_clauses": (
                int(args.max_entry_clauses) if args.max_entry_clauses is not None else None
            ),
            "max_total_conds": (
                int(args.max_total_conds) if args.max_total_conds is not None else None
            ),
            "max_unique_atoms": (
                int(args.max_unique_atoms) if args.max_unique_atoms is not None else None
            ),
        },
        "out_dir": str(out_dir),
    }
    (out_dir / "batch_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print(f"\nBatch complete. Output: {out_dir}")
    print(f"Configs: {meta['configs_total']}  Errors: {meta['errors_total']}")
    print(f"Sweep:   {out_dir / 'results.csv'}")
    print(f"Full:    {out_dir / 'results_full.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())