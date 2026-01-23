# tools/make_grid.py
from __future__ import annotations

import argparse
import itertools
import json
import math
import copy
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def _canonical_json(obj: Any) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _jsonl_write(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(_canonical_json(row))
            f.write("\n")
            n += 1
    return n


def _ladder(
    tp_r: List[float],
    tp_f: List[float],
    move_to_be_at_r: Optional[float],
) -> Tuple[List[float], List[float], Optional[float]]:
    if len(tp_r) != len(tp_f):
        raise ValueError("tp_r and tp_f must be same length")
    if any((not math.isfinite(x) or x <= 0) for x in tp_r):
        raise ValueError("tp_r must be positive finite")
    if any((not math.isfinite(x) or x <= 0 or x > 1) for x in tp_f):
        raise ValueError("tp_f must be in (0,1]")
    if move_to_be_at_r is not None and (not math.isfinite(move_to_be_at_r)):
        raise ValueError("move_to_be_at_r must be finite or None")
    return tp_r, tp_f, move_to_be_at_r


def _cfg_key(cfg: Dict[str, Any]) -> str:
    """
    Stable dedupe key that ignores 'strategy_name' so we don't treat identical
    strategies as different just because the name differs.
    """
    c = dict(cfg)
    c.pop("strategy_name", None)
    return _canonical_json(c)


def _dedupe_preserve_order(cfgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    for cfg in cfgs:
        k = _cfg_key(cfg)
        if k in seen:
            continue
        seen.add(k)
        out.append(cfg)
    return out

# ============================================================
# Legacy space (v1): entry_conditions AND-only
# ============================================================


def build_space() -> List[Dict[str, Any]]:
    """
    Broad search space using your current UniversalStrategy capabilities:
    - Entry: RSI threshold
    - Filters: trend (ema_200 / ema_50) optional, plus optional vol filter via atr_pct
    - Stops: ATR or PCT
    - TP ladders: a few presets
    - Sizing: risk_per_trade_pct  max_leverage
    """

    # Entry axis
    rsi_thresholds = [30.0, 35.0, 40.0, 45.0, 50.0]

    # Filter axis (trend)
    trend_filters: List[List[Dict[str, Any]]] = [
        # none
        [],
        # ema200 only
        [
            {
                "indicator": "close",
                "operator": ">",
                "threshold": 0.0,
                "ref_indicator": "ema_200",
            }
        ],
        # ema200  ema50
        [
            {
                "indicator": "close",
                "operator": ">",
                "threshold": 0.0,
                "ref_indicator": "ema_200",
            },
            {
                "indicator": "close",
                "operator": ">",
                "threshold": 0.0,
                "ref_indicator": "ema_50",
            },
        ],
    ]

    # Optional vol filter axis (atr_pct is a feature you compute)
    vol_filters: List[List[Dict[str, Any]]] = [
        [],
        [{"indicator": "atr_pct", "operator": "<", "threshold": 0.8}],
        [{"indicator": "atr_pct", "operator": "<", "threshold": 1.2}],
        [{"indicator": "atr_pct", "operator": "<", "threshold": 1.6}],
    ]

    # Stop models
    stop_models: List[Tuple[str, float]] = [
        ("ATR", 1.5),
        ("ATR", 2.0),
        ("ATR", 2.5),
        ("ATR", 3.0),
        ("PCT", 0.005),
        ("PCT", 0.010),
        ("PCT", 0.015),
    ]

    # TP ladders
    ladders = [
        _ladder([1.5, 3.0], [0.5, 1.0], 1.5),
        _ladder([2.0, 4.0], [0.5, 1.0], 2.0),
        _ladder([2.0, 5.0], [0.5, 1.0], 2.0),
        _ladder([1.0, 2.0], [0.5, 1.0], 1.0),
        _ladder([1.5, 4.0], [0.33, 1.0], 1.5),
        _ladder([2.0, 6.0], [0.33, 1.0], 2.0),
        _ladder([2.0, 5.0], [0.5, 1.0], None),
    ]

    # Risk sizing axis
    risk_pcts = [0.005, 0.01]
    leverages = [2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0]

    sides = ["long", "short"]

    space: List[Dict[str, Any]] = []
    for (
        side,
        rsi_thr,
        trend,
        volf,
        (sl_type, sl_param),
        (tp_r, tp_f, be_at),
        risk_pct,
        max_lev,
    ) in itertools.product(
        sides,
        rsi_thresholds,
        trend_filters,
        vol_filters,
        stop_models,
        ladders,
        risk_pcts,
        leverages,
    ):
        cfg = {
            "strategy_name": (
                f"{side}_rsi{int(rsi_thr)}_{sl_type.lower()}{sl_param}_"
                f"tp{len(tp_r)}_lev{max_lev}_rp{risk_pct}"
            ),
            "side": side,
            "entry_conditions": [
                {"indicator": "rsi_14", "operator": "<", "threshold": rsi_thr}
            ],
            "filters": [*trend, *volf],
            "risk": {
                "risk_per_trade_pct": risk_pct,
                "max_leverage": max_lev,
                "sl_type": sl_type,
                "sl_param": sl_param,
                "tp_r_multiples": tp_r,
                "tp_is_market": True,
                "tp_fractions": tp_f,
                "move_to_be_at_r": be_at,
            },
        }
        space.append(cfg)

    return space

# ============================================================
# Option 2 space (DNF-lite): entry_any = OR of AND clauses
# ============================================================


MAX_ENTRY_ANY_CLAUSES = 3
MAX_ENTRY_ANY_CONDS_PER_CLAUSE = 6

Archetype = Literal[
    "pullback",
    "breakout",
    "squeeze_breakout",
    "mixed",
]

CLAUSE_LEN_ITEMS = [2, 3, 4]
CLAUSE_LEN_WEIGHTS = [0.55, 0.35, 0.10]

def _weighted_choice(rng: random.Random, items: List[Any], weights: List[float]) -> Any:
    if not items:
        return None
    if len(items) != len(weights) or sum(weights) <= 0:
        return rng.choice(items)
    x = rng.random() * float(sum(weights))
    s = 0.0
    for it, w in zip(items, weights):
        s += float(w)
        if x <= s:
            return it
    return items[-1]

def _sample_clause_len(
    rng: random.Random,
    *,
    min_conds: int,
    max_conds: int,
) -> int:
    lo = int(max(1, min_conds))
    hi = int(max(lo, max_conds))
    n = int(_weighted_choice(rng, CLAUSE_LEN_ITEMS, CLAUSE_LEN_WEIGHTS))
    return int(max(lo, min(hi, n)))

def _op_dir(op: str) -> str:
    """
    Bucket operators by direction so we can avoid contradictions inside a clause.
    """
    if op in (">", ">="):
        return "gt"
    if op in ("<", "<="):
        return "lt"
    return "other"


def _cond_key(c: Dict[str, Any]) -> str:
    # Stable key to prevent duplicates inside a clause.
    return _canonical_json(
        {
            "indicator": c.get("indicator"),
            "operator": c.get("operator"),
            "threshold": c.get("threshold"),
            "ref_indicator": c.get("ref_indicator", None),
        }
    )

def _clause_sig(clause: List[Dict[str, Any]]) -> str:
    """
    Stable signature for a clause (ignores ordering of conditions).
    Used to prevent duplicate clauses inside entry_any.
    """
    keys = sorted(_cond_key(c) for c in clause)
    return _canonical_json(keys)


def _atom_key(c: Dict[str, Any]) -> str:
    """
    A condition "atom" is the left indicator plus optional ref_indicator.
    This lets us prevent contradictions like close > ema_50 AND close < ema_50.
    """
    ind = str(c.get("indicator", ""))
    ref = c.get("ref_indicator", None)
    return f"{ind}|{str(ref) if ref is not None else ''}"
    
def _pick_unique(
    rng: random.Random,
    pool: List[Dict[str, Any]],
    used: Set[str],
    used_atom_dirs: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    if not pool:
        return None
    # Try a few times to find a not-yet-used condition (avoid duplicates).
    for _ in range(12):
        c = dict(rng.choice(pool))
        k = _cond_key(c)
        if k in used:
            continue
        atom = _atom_key(c)
        d = _op_dir(str(c.get("operator", "")))
        prev = used_atom_dirs.get(atom)
        if prev is not None and prev != d:
            # Contradiction (same atom, opposite direction)
            continue

        used.add(k)
        used_atom_dirs[atom] = d
        return c
    return None


def _build_clause_long(
    rng: random.Random,
    *,
    min_conds: int,
    max_conds: int,
    mode: str = "selective",
) -> List[Dict[str, Any]]:
    """
    Build one AND-clause for long entries.

    v2 goals:
      - enforce the hard trend gate
      - allow short clauses (2-3) for cadence
      - avoid default tail triggers (e.g. donch_pos > 0.8) that kill frequency
    mode:
      - "loose": cadence-friendly opportunity clause
      - "selective": tighter clause (still bounded, avoids extremes by default)
    """
    # Discrete thresholds keep configs stable  dedupe-friendly.
    rsi_low = [40.0, 45.0, 50.0]
    rsi_high = [50.0, 55.0, 60.0]
    atr_caps = [1.2, 1.6, 2.0]
    rvol_min = [1.0, 1.2, 1.5]

    # Context/regime bins (15m/1h)
    adx_min = [15.0, 20.0, 25.0]
    bbw_max = [0.06, 0.08, 0.10]
    donch_pullback = [0.30, 0.40, 0.45]
    donch_breakout = [0.55, 0.60, 0.70]
    donch_periods = [20, 55]


    gate_pool = [
       {"indicator": "trend_up_1h", "operator": ">", "threshold": 0.5}
    ]


    # --- Condition pools ---
    trend_pool = [
        {
            "indicator": "close",
            "operator": ">",
            "threshold": 0.0,
            "ref_indicator": "ema_200",
        },
        {
            "indicator": "close",
            "operator": ">",
            "threshold": 0.0,
            "ref_indicator": "ema_50",
        },
        {
            "indicator": "ema_50",
            "operator": ">",
            "threshold": 0.0,
            "ref_indicator": "ema_200",
        },
    ]

    # Context regime pools (15m/1h only) - moderate bins
    trend_strength_pool = [
        {"indicator": "adx_14_1h", "operator": ">", "threshold": float(rng.choice(adx_min))},
        {"indicator": "adx_14_15m", "operator": ">", "threshold": float(rng.choice(adx_min))},
    ]
    vol_regime_pool = [
        {"indicator": "bb_width_20_1h", "operator": "<", "threshold": float(rng.choice(bbw_max))},
        {"indicator": "bb_width_20_15m", "operator": "<", "threshold": float(rng.choice(bbw_max))},
    ]
    # Structure (15m Donchian): use 20 or 55 randomly per condition
    p = int(rng.choice(donch_periods))
    structure_pool = [
        {
            "indicator": f"donch_pos_{p}_15m",
            "operator": "<",
            "threshold": float(rng.choice(donch_pullback)),
        },
        {
            "indicator": f"donch_pos_{p}_15m",
            "operator": ">",
            "threshold": float(rng.choice(donch_breakout)),
        },
    ]

    # v2: explicitly add a mild "opportunity" pool for cadence
    loose_trigger_pool = [
        {"indicator": "close", "operator": "<", "threshold": 0.0, "ref_indicator": "ema_20"},
        {"indicator": "rsi_14", "operator": "<", "threshold": float(rng.choice(rsi_low))},
        {"indicator": "bb_z_20_15m", "operator": "<", "threshold": -0.5},
    ]
    
    mom_pool = [
        {
            "indicator": "rsi_14",
            "operator": "<",
            "threshold": float(rng.choice(rsi_low)),
        },
        {
            "indicator": "rsi_14",
            "operator": ">",
            "threshold": float(rng.choice(rsi_high)),
        },
    ]

    vol_pool = [
        {
            "indicator": "atr_pct",
            "operator": "<",
            "threshold": float(rng.choice(atr_caps)),
        }
    ]

    volu_pool = [
        {
            "indicator": "rvol_50",
            "operator": ">",
            "threshold": float(rng.choice(rvol_min)),
        }
    ]

    mr_pool = [
        {
            "indicator": "close",
            "operator": "<",
            "threshold": 0.0,
            "ref_indicator": "ema_20",
        },
        {
            "indicator": "close",
            "operator": "<",
            "threshold": 0.0,
            "ref_indicator": "ema_50",
        },
    ]

    # --- Clause construction with bounded complexity ---
    used: Set[str] = set()
    used_atom_dirs: Dict[str, str] = {}
    clause: List[Dict[str, Any]] = []

    target_n = _sample_clause_len(rng, min_conds=min_conds, max_conds=max_conds)
    target_n = min(int(target_n), MAX_ENTRY_ANY_CONDS_PER_CLAUSE)

    # Always include the hard gate
    c = _pick_unique(rng, gate_pool, used, used_atom_dirs)
    if c is not None:
        clause.append(c)

    if mode.strip().lower() == "loose":
        # Gate + 1 trigger (+ optional 1 mild regime if target_n allows)
        c = _pick_unique(rng, loose_trigger_pool, used, used_atom_dirs)
        if c is not None:
            clause.append(c)
        mild_regime_pool = vol_pool + trend_strength_pool + vol_regime_pool
        while len(clause) < target_n:
            c = _pick_unique(rng, mild_regime_pool, used, used_atom_dirs)
            if c is None:
                break
            clause.append(c)
        return clause

    # Selective: keep a trend anchor and one trigger, then optional regime/structure
    c = _pick_unique(rng, trend_pool, used, used_atom_dirs)
    if c is not None:
        clause.append(c)

    trigger_pool = mom_pool + mr_pool
    c = _pick_unique(rng, trigger_pool, used, used_atom_dirs)
    if c is not None:
        clause.append(c)

    regime_pool = vol_pool + volu_pool + trend_strength_pool + vol_regime_pool + structure_pool
    any_pool = trigger_pool + regime_pool

    while len(clause) < target_n:
        c = _pick_unique(rng, any_pool, used, used_atom_dirs)
        if c is None:
            break
        clause.append(c)

    return clause

def _build_clause_breakout_long(
    rng: random.Random,
    *,
    min_conds: int,
    max_conds: int,
    mode: str = "selective",
) -> List[Dict[str, Any]]:
    """
    Trend breakout continuation (long):
      - Gate: trend_up_1h
      - Trigger: close > donch_hi_{20|55}_15m
      - Optional confirms: rvol/adx
    """
    used: Set[str] = set()
    used_atom_dirs: Dict[str, str] = {}
    clause: List[Dict[str, Any]] = []

    target_n = _sample_clause_len(rng, min_conds=min_conds, max_conds=max_conds)
    target_n = min(int(target_n), MAX_ENTRY_ANY_CONDS_PER_CLAUSE)

    p = int(rng.choice([20, 55]))
    gate_pool = [{"indicator": "trend_up_1h", "operator": ">", "threshold": 0.5}]
    trigger_pool = [
        {
            "indicator": "close",
            "operator": ">",
            "threshold": 0.0,
            "ref_indicator": f"donch_hi_{p}_15m",
        }
    ]

    confirm_pool = [
        {"indicator": "rvol_50", "operator": ">", "threshold": float(rng.choice([1.0, 1.2, 1.5]))},
        {"indicator": "adx_14_15m", "operator": ">", "threshold": float(rng.choice([15.0, 20.0]))},
        {"indicator": "adx_14_1h", "operator": ">", "threshold": float(rng.choice([15.0, 20.0]))},
        {"indicator": "rsi_14", "operator": ">", "threshold": float(rng.choice([50.0, 55.0, 60.0]))},
    ]
    c = _pick_unique(rng, gate_pool, used, used_atom_dirs)
    if c is not None:
        clause.append(c)

    c = _pick_unique(rng, trigger_pool, used, used_atom_dirs)
    if c is not None:
        clause.append(c)

    if mode.strip().lower() == "loose":
        return clause

    while len(clause) < target_n:
        c = _pick_unique(rng, confirm_pool, used, used_atom_dirs)
        if c is None:
            break
        clause.append(c)

    return clause

def _build_clause_breakout_short(
    rng: random.Random,
    *,
    min_conds: int,
    max_conds: int,
    mode: str = "selective",
) -> List[Dict[str, Any]]:
    """
    Trend breakout continuation (short):
      - Gate: trend_down_1h
      - Trigger: close < donch_lo_{20|55}_15m
      - Optional confirms: rvol/adx
    """
    used: Set[str] = set()
    used_atom_dirs: Dict[str, str] = {}
    clause: List[Dict[str, Any]] = []

    target_n = _sample_clause_len(rng, min_conds=min_conds, max_conds=max_conds)
    target_n = min(int(target_n), MAX_ENTRY_ANY_CONDS_PER_CLAUSE)

    p = int(rng.choice([20, 55]))
    gate_pool = [{"indicator": "trend_down_1h", "operator": ">", "threshold": 0.5}]
    trigger_pool = [
        {
            "indicator": "close",
            "operator": "<",
            "threshold": 0.0,
            "ref_indicator": f"donch_lo_{p}_15m",
        }
    ]
    confirm_pool = [
        {"indicator": "rvol_50", "operator": ">", "threshold": float(rng.choice([1.0, 1.2, 1.5]))},
        {"indicator": "adx_14_15m", "operator": ">", "threshold": float(rng.choice([15.0, 20.0]))},
        {"indicator": "adx_14_1h", "operator": ">", "threshold": float(rng.choice([15.0, 20.0]))},
        {"indicator": "rsi_14", "operator": "<", "threshold": float(rng.choice([50.0, 45.0, 40.0]))},
    ]

    c = _pick_unique(rng, gate_pool, used, used_atom_dirs)
    if c is not None:
        clause.append(c)

    c = _pick_unique(rng, trigger_pool, used, used_atom_dirs)
    if c is not None:
        clause.append(c)
    if mode.strip().lower() == "loose":
        return clause

    while len(clause) < target_n:
        c = _pick_unique(rng, confirm_pool, used, used_atom_dirs)
        if c is None:
            break
        clause.append(c)

    return clause

def _build_clause_squeeze_breakout_long(
    rng: random.Random,
    *,
    min_conds: int,
    max_conds: int,
    mode: str = "selective",
) -> List[Dict[str, Any]]:
    """
    Squeeze -> breakout (long):
      - Gate: trend_up_1h
      - Setup: bb_width_20_1h < threshold
      - Trigger: close > donch_hi_{20|55}_15m
    """
    used: Set[str] = set()
    used_atom_dirs: Dict[str, str] = {}
    clause: List[Dict[str, Any]] = []

    target_n = _sample_clause_len(rng, min_conds=min_conds, max_conds=max_conds)
    target_n = min(int(target_n), MAX_ENTRY_ANY_CONDS_PER_CLAUSE)

    p = int(rng.choice([20, 55]))
    bbw = float(rng.choice([0.06, 0.08, 0.10, 0.12]))

    gate_pool = [{"indicator": "trend_up_1h", "operator": ">", "threshold": 0.5}]
    setup_pool = [{"indicator": "bb_width_20_1h", "operator": "<", "threshold": bbw}]
    trigger_pool = [
        {
            "indicator": "close",
            "operator": ">",
            "threshold": 0.0,
            "ref_indicator": f"donch_hi_{p}_15m",
        }
    ]
    confirm_pool = [
        {"indicator": "rvol_50", "operator": ">", "threshold": float(rng.choice([1.0, 1.2, 1.5]))},
        {"indicator": "adx_14_1h", "operator": ">", "threshold": float(rng.choice([15.0, 20.0]))},
    ]
    for pool in (gate_pool, setup_pool, trigger_pool):
        c = _pick_unique(rng, pool, used, used_atom_dirs)
        if c is not None:
            clause.append(c)
    if mode.strip().lower() == "loose":
        return clause

    while len(clause) < target_n:
        c = _pick_unique(rng, confirm_pool, used, used_atom_dirs)
        if c is None:
            break
        clause.append(c)

    return clause

def _build_clause_squeeze_breakout_short(
    rng: random.Random,
    *,
    min_conds: int,
    max_conds: int,
    mode: str = "selective",
) -> List[Dict[str, Any]]:
    """
    Squeeze -> breakout (short):
      - Gate: trend_down_1h
      - Setup: bb_width_20_1h < threshold
      - Trigger: close < donch_lo_{20|55}_15m
    """
    used: Set[str] = set()
    used_atom_dirs: Dict[str, str] = {}
    clause: List[Dict[str, Any]] = []

    target_n = _sample_clause_len(rng, min_conds=min_conds, max_conds=max_conds)
    target_n = min(int(target_n), MAX_ENTRY_ANY_CONDS_PER_CLAUSE)

    p = int(rng.choice([20, 55]))
    bbw = float(rng.choice([0.06, 0.08, 0.10, 0.12]))

    gate_pool = [{"indicator": "trend_down_1h", "operator": ">", "threshold": 0.5}]
    setup_pool = [{"indicator": "bb_width_20_1h", "operator": "<", "threshold": bbw}]
    trigger_pool = [
        {
            "indicator": "close",
            "operator": "<",
            "threshold": 0.0,
            "ref_indicator": f"donch_lo_{p}_15m",
        }
    ]
    confirm_pool = [
        {"indicator": "rvol_50", "operator": ">", "threshold": float(rng.choice([1.0, 1.2, 1.5]))},
        {"indicator": "adx_14_1h", "operator": ">", "threshold": float(rng.choice([15.0, 20.0]))},
    ]
    for pool in (gate_pool, setup_pool, trigger_pool):
        c = _pick_unique(rng, pool, used, used_atom_dirs)
        if c is not None:
            clause.append(c)
    if mode.strip().lower() == "loose":
        return clause

    while len(clause) < target_n:
        c = _pick_unique(rng, confirm_pool, used, used_atom_dirs)
        if c is None:
            break
        clause.append(c)

    return clause

def _build_clause_short(
    rng: random.Random,
    *,
    min_conds: int,
    max_conds: int,
    mode: str = "selective",
) -> List[Dict[str, Any]]:
    """
    Build one AND-clause for short entries.

    v2 goals mirror long:
      - always include the hard trend_down gate
      - allow 2-3 condition clauses for cadence
      - avoid default tail triggers

    mode: "loose" or "selective"
    """
    rsi_low = [40.0, 45.0, 50.0]
    rsi_high = [50.0, 55.0, 60.0]
    atr_caps = [1.2, 1.6, 2.0]
    rvol_min = [1.0, 1.2, 1.5]

    adx_min = [15.0, 20.0, 25.0]
    bbw_max = [0.06, 0.08, 0.10]
    donch_pullback = [0.55, 0.60, 0.70]
    donch_breakdown = [0.30, 0.40, 0.45]
    donch_periods = [20, 55]

    gate_pool = [
        {"indicator": "trend_down_1h", "operator": ">", "threshold": 0.5}
    ]


    # Bear trend pool (mirrors long)
    trend_pool = [
        {
            "indicator": "close",
            "operator": "<",
            "threshold": 0.0,
            "ref_indicator": "ema_200",
        },
        {
            "indicator": "close",
            "operator": "<",
            "threshold": 0.0,
            "ref_indicator": "ema_50",
        },
        {
            "indicator": "ema_50",
            "operator": "<",
            "threshold": 0.0,
            "ref_indicator": "ema_200",
        },
    ]

    # Momentum pool: allow both to discover which works for shorts
    mom_pool = [
        {
            "indicator": "rsi_14",
            "operator": "<",
            "threshold": float(rng.choice(rsi_low)),
        },
        {
            "indicator": "rsi_14",
            "operator": ">",
            "threshold": float(rng.choice(rsi_high)),
        },
    ]

    vol_pool = [
        {
            "indicator": "atr_pct",
            "operator": "<",
            "threshold": float(rng.choice(atr_caps)),
        }
    ]

    volu_pool = [
        {
            "indicator": "rvol_50",
            "operator": ">",
            "threshold": float(rng.choice(rvol_min)),
        }
    ]
    # Mean reversion for shorts: pullback up before shorting
    mr_pool = [
        {
            "indicator": "close",
            "operator": ">",
            "threshold": 0.0,
            "ref_indicator": "ema_20",
        },
        {
            "indicator": "close",
            "operator": ">",
            "threshold": 0.0,
            "ref_indicator": "ema_50",
        },
    ]

    loose_trigger_pool = [
        {"indicator": "close", "operator": ">", "threshold": 0.0, "ref_indicator": "ema_20"},
        {"indicator": "rsi_14", "operator": ">", "threshold": 50.0},
        {"indicator": "bb_z_20_15m", "operator": ">", "threshold": 0.5},
    ]

    trend_strength_pool = [
        {"indicator": "adx_14_1h", "operator": ">", "threshold": float(rng.choice(adx_min))},
        {"indicator": "adx_14_15m", "operator": ">", "threshold": float(rng.choice(adx_min))},
    ]
    vol_regime_pool = [
        {"indicator": "bb_width_20_1h", "operator": "<", "threshold": float(rng.choice(bbw_max))},
        {"indicator": "bb_width_20_15m", "operator": "<", "threshold": float(rng.choice(bbw_max))},
    ]
    p = int(rng.choice(donch_periods))
    structure_pool = [
        {
            "indicator": f"donch_pos_{p}_15m",
            "operator": ">",
            "threshold": float(rng.choice(donch_pullback)),
        },
        {
            "indicator": f"donch_pos_{p}_15m",
            "operator": "<",
            "threshold": float(rng.choice(donch_breakdown)),
        },
    ]

    used: Set[str] = set()
    used_atom_dirs: Dict[str, str] = {}
    clause: List[Dict[str, Any]] = []

    target_n = _sample_clause_len(rng, min_conds=min_conds, max_conds=max_conds)
    target_n = min(int(target_n), MAX_ENTRY_ANY_CONDS_PER_CLAUSE)

    # Always include the hard gate
    c = _pick_unique(rng, gate_pool, used, used_atom_dirs)
    if c is not None:
        clause.append(c)

    if mode.strip().lower() == "loose":
        c = _pick_unique(rng, loose_trigger_pool, used, used_atom_dirs)
        if c is not None:
            clause.append(c)
        mild_regime_pool = vol_pool + trend_strength_pool + vol_regime_pool
        while len(clause) < target_n:
            c = _pick_unique(rng, mild_regime_pool, used, used_atom_dirs)
            if c is None:
                break
            clause.append(c)
        return clause

    c = _pick_unique(rng, trend_pool, used, used_atom_dirs)
    if c is not None:
        clause.append(c)

    trigger_pool = mom_pool + mr_pool
    c = _pick_unique(rng, trigger_pool, used, used_atom_dirs)
    if c is not None:
        clause.append(c)

    regime_pool = (
        vol_pool + volu_pool + trend_strength_pool + vol_regime_pool + structure_pool
    )
    any_pool = trigger_pool + regime_pool

    while len(clause) < target_n:
        c = _pick_unique(rng, any_pool, used, used_atom_dirs)
        if c is None:
            break
        clause.append(c)

    return clause

def _build_clause_by_archetype(
    rng: random.Random,
    *,
    archetype: Archetype,
    side: str,
    min_conds: int,
    max_conds: int,
    mode: str,
) -> List[Dict[str, Any]]:
    side = str(side).lower()
    if archetype == "pullback":
        if side == "long":
            return _build_clause_long(
                rng,
                min_conds=min_conds,
                max_conds=max_conds,
                mode=mode,
            )
        return _build_clause_short(
            rng,
            min_conds=min_conds,
            max_conds=max_conds,
            mode=mode,
        )

    if archetype == "breakout":
        if side == "long":
            return _build_clause_breakout_long(
                rng,
                min_conds=min_conds,
                max_conds=max_conds,
                mode=mode,
            )
        return _build_clause_breakout_short(
            rng,
            min_conds=min_conds,
            max_conds=max_conds,
            mode=mode,
        )

    if archetype == "squeeze_breakout":
        if side == "long":
            return _build_clause_squeeze_breakout_long(
                rng,
                min_conds=min_conds,
                max_conds=max_conds,
                mode=mode,
            )
        return _build_clause_squeeze_breakout_short(
            rng,
            min_conds=min_conds,
            max_conds=max_conds,
            mode=mode,
        )

    raise ValueError(f"Unknown archetype: {archetype}")


def build_space_entry_any(
    *,
    max_configs: int,
    seed: int,
    exclude_keys: Optional[Set[str]] = None,
    archetype: Archetype = "mixed",
) -> List[Dict[str, Any]]:
    """
    Generates a bounded Option 2 search space using sampling (Mode B).

    Output configs use:
      - entry_any: OR-of-AND clauses (DNF-lite)
      - filters: empty (entry logic handles gating for now)
      - risk: same structure as legacy

    Deterministic by seed.
    """
    if exclude_keys is None:
        exclude_keys = set()

    rng = random.Random(int(seed))

    archetype = str(archetype).strip().lower()
    if archetype not in {"pullback", "breakout", "squeeze_breakout", "mixed"}:
        raise ValueError(f"Invalid archetype: {archetype}")

    # Risk sizing axis (same as legacy for now)
    risk_pcts = [0.005, 0.01]
    leverages = [2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0]

    # Stop models
    stop_models: List[Tuple[str, float]] = [
        ("ATR", 1.5),
        ("ATR", 2.0),
        ("ATR", 2.5),
        ("ATR", 3.0),
        ("PCT", 0.005),
        ("PCT", 0.010),
        ("PCT", 0.015),
    ]

    ladders = [
        _ladder([1.5, 3.0], [0.5, 1.0], 1.5),
        _ladder([2.0, 4.0], [0.5, 1.0], 2.0),
        _ladder([2.0, 5.0], [0.5, 1.0], 2.0),
        _ladder([1.0, 2.0], [0.5, 1.0], 1.0),
        _ladder([1.5, 4.0], [0.33, 1.0], 1.5),
        _ladder([2.0, 6.0], [0.33, 1.0], 2.0),
        _ladder([2.0, 5.0], [0.5, 1.0], None),
    ]

    sides = ["long", "short"]

    out: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = int(max(500, max_configs * 50))

    while len(out) < int(max_configs) and attempts < max_attempts:
        attempts += 1

        side = str(rng.choice(sides))
        n_clauses = int(_weighted_choice(rng, [1, 2, 3], [0.10, 0.65, 0.25]))

        entry_any: List[List[Dict[str, Any]]] = []
        seen_clause_sigs: Set[str] = set()
        # Clause archetype policy:
        # - If archetype is fixed: use that archetype for all clauses.
        # - If mixed: first clause is pullback(loose) for cadence, later clauses are
        #   sampled from archetypes (pullback/breakout/squeeze_breakout).
        first_mode = "loose"
        for ci in range(n_clauses):
            clause_added = False
            mode = first_mode if ci == 0 else "selective"
            for _try in range(25):
                if archetype == "mixed":
                    if ci == 0:
                        arch_ci: Archetype = "pullback"
                    else:
                        arch_ci = _weighted_choice(
                            rng,
                            ["pullback", "breakout", "squeeze_breakout"],
                            [0.40, 0.40, 0.20],
                        )
                else:
                    arch_ci = archetype  # type: ignore[assignment]

                clause = _build_clause_by_archetype(
                    rng,
                    archetype=arch_ci,  # type: ignore[arg-type]
                    side=side,
                    min_conds=2,
                    max_conds=4,
                    mode=mode,
                )
                sig = _clause_sig(clause)
                if sig in seen_clause_sigs:
                    continue
                seen_clause_sigs.add(sig)
                entry_any.append(clause)
                clause_added = True
                break

            if not clause_added:
                # Could not find a unique clause quickly; stop early.
                break

        if not entry_any:
            continue

        (sl_type, sl_param) = rng.choice(stop_models)
        (tp_r, tp_f, be_at) = rng.choice(ladders)
        risk_pct = float(rng.choice(risk_pcts))
        max_lev = float(rng.choice(leverages))

        cfg = {
            "strategy_name": (
                f"DNF_{side}_{archetype}_c{n_clauses}_lev{max_lev}_rp{risk_pct}"
            ),
            "side": side,
            "entry_any": entry_any,
            "filters": [],
            "risk": {
                "sizing_mode": "dynamic_leverage",
                "leverage_min": 2.0,
                "leverage_max": 15.0,
                "risk_max_pct": 0.05,
                "conf_model": rng.choice(["compression", "expansion"]),
                "risk_per_trade_pct": risk_pct,
                "max_leverage": max_lev,
                "sl_type": str(sl_type),
                "sl_param": float(sl_param),
                "tp_r_multiples": tp_r,
                "tp_is_market": True,
                "tp_fractions": tp_f,
                "move_to_be_at_r": be_at,
            },
        }

        k = _cfg_key(cfg)
        if k in exclude_keys:
            continue
        exclude_keys.add(k)
        out.append(cfg)

    return out



def sample_space(
    space: List[Dict[str, Any]],
    max_configs: int,
    seed: int,
    exclude_keys: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    if exclude_keys is None:
        exclude_keys = set()

    candidates = [cfg for cfg in space if _cfg_key(cfg) not in exclude_keys]
    if max_configs <= 0 or max_configs >= len(candidates):
        return candidates

    rng = random.Random(int(seed))
    idxs = list(range(len(candidates)))
    rng.shuffle(idxs)
    keep = idxs[:max_configs]
    return [candidates[i] for i in keep]


def _load_winner_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Winner config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("Winner config.json must be a JSON object")
    return obj


def build_neighborhood_from_winner(
    winner_cfg: Dict[str, Any],
    *,
    include_ema50_toggle: bool = False,
) -> List[Dict[str, Any]]:
    """
    Tight neighborhood search around the winner.

    We keep the overall "shape" of the winner, but vary key thresholds
    to test stability.
    """
    # Defaults from your observed winner shape
    side = str(winner_cfg.get("side", "long"))
    risk = dict(winner_cfg.get("risk", {}))
    base_risk_pct = float(risk.get("risk_per_trade_pct", 0.005))
    base_max_lev = float(risk.get("max_leverage", 5.0))

    # Neighborhood axes (tuned to your winner)
    rsi_thresholds = [30.0, 32.5, 35.0, 37.5, 40.0]
    atr_pct_caps = [1.2, 1.4, 1.6, 1.8]
    stop_pcts = [0.0075, 0.0100, 0.0125]
    tp2_rs = [3.0, 4.0, 5.0]
    move_be = [1.0, 1.5, 2.0]

    # Base filters: must include ema_200 trend filter
    # If winner_cfg already has filters, we will keep only "close > ema_200" and
    # rebuild atr_pct filter from our axis.
    base_filters: List[Dict[str, Any]] = []
    for f in (winner_cfg.get("filters") or []):
        if not isinstance(f, dict):
            continue
        if (
            f.get("indicator") == "close"
            and f.get("operator") == ">"
            and f.get("ref_indicator") == "ema_200"
        ):
            base_filters.append(
                {
                    "indicator": "close",
                    "operator": ">",
                    "threshold": 0.0,
                    "ref_indicator": "ema_200",
                }
            )

    if not base_filters:
        base_filters = [
            {
                "indicator": "close",
                "operator": ">",
                "threshold": 0.0,
                "ref_indicator": "ema_200",
            }
        ]

    ema50_filter = {
        "indicator": "close",
        "operator": ">",
        "threshold": 0.0,
        "ref_indicator": "ema_50",
    }

    filters_variants: List[List[Dict[str, Any]]] = []
    if include_ema50_toggle:
        filters_variants = [
            base_filters,
            [*base_filters, ema50_filter],
        ]
    else:
        filters_variants = [base_filters]

    out: List[Dict[str, Any]] = []
    for (
        rsi_thr,
        atr_cap,
        stop_pct,
        tp2,
        be_at,
        filt_base,
    ) in itertools.product(
        rsi_thresholds,
        atr_pct_caps,
        stop_pcts,
        tp2_rs,
        move_be,
        filters_variants,
    ):
        cfg = {
            "strategy_name": (
                f"NEIGH_{side}_rsi{rsi_thr}_atrpct{atr_cap}_"
                f"pct{stop_pct}_tp1.5_{tp2}_be{be_at}"
            ),
            "side": side,
            "entry_conditions": [
                {"indicator": "rsi_14", "operator": "<", "threshold": rsi_thr}
            ],
            "filters": [
                *filt_base,
                {"indicator": "atr_pct", "operator": "<", "threshold": atr_cap},
            ],
            "risk": {
                "risk_per_trade_pct": base_risk_pct,
                "max_leverage": base_max_lev,
                "sl_type": "PCT",
                "sl_param": stop_pct,
                "tp_r_multiples": [1.5, tp2],
                "tp_is_market": True,
                "tp_fractions": [0.33, 1.0],
                "move_to_be_at_r": be_at,
            },
        }
        out.append(cfg)

    return out

def _neighbor_thresholds(ind: str, op: str, thr: float) -> List[float]:
    """
    Discrete neighborhood bins per indicator family.
    Returns a small set containing the original threshold plus nearby sensible values.
    """
    ind = str(ind)
    op = str(op)
    try:
        thr_f = float(thr)
    except Exception:
        return [thr]

    def around(values: List[float]) -> List[float]:
        # keep values within list that are "near" the current threshold
        vals = sorted(set(float(x) for x in values))
        if not vals:
            return [thr_f]
        # pick up to 5 closest bins
        vals = sorted(vals, key=lambda x: abs(x - thr_f))[:5]
        vals.append(thr_f)
        return sorted(set(vals))

    # RSI family
    if ind.startswith("rsi_"):
        return around([25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75])

    # Relative volume
    if ind.startswith("rvol_"):
        return around([0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5])

    # ADX
    if ind.startswith("adx_"):
        return around([10, 12, 15, 18, 20, 22, 25, 30, 35])
    # Bollinger width (roughly: 0.03–0.12 typical)
    if ind.startswith("bb_width_"):
        return around([0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12])

    # Donchian position: suggest bins based on operator direction
    if ind.startswith("donch_pos_"):
        if op in ("<", "<="):
            return around([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
        return around([0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85])
    # Default: small neighborhood around the value
    # (only if it's reasonable magnitude)
    if abs(thr_f) < 10:
        return sorted(set([thr_f, thr_f * 0.9, thr_f * 1.1]))
    return [thr_f]


def _maybe_swap_donch_period(ind: str, *, p_swap: float, rng: random.Random) -> str:
    """
    Swap donch_pos_20_15m <-> donch_pos_55_15m (and hi/lo equivalents) with probability.
    """
    if rng.random() > float(p_swap):
        return ind
    s = str(ind)
    if "donch_pos_20_15m" in s:
        return s.replace("donch_pos_20_15m", "donch_pos_55_15m")
    if "donch_pos_55_15m" in s:
        return s.replace("donch_pos_55_15m", "donch_pos_20_15m")
    if "donch_hi_20_15m" in s:
        return s.replace("donch_hi_20_15m", "donch_hi_55_15m")
    if "donch_hi_55_15m" in s:
        return s.replace("donch_hi_55_15m", "donch_hi_20_15m")
    if "donch_lo_20_15m" in s:
        return s.replace("donch_lo_20_15m", "donch_lo_55_15m")
    if "donch_lo_55_15m" in s:
        return s.replace("donch_lo_55_15m", "donch_lo_20_15m")
    return ind


def build_neighborhood_from_winner_entry_any(
    winner_cfg: Dict[str, Any],
    *,
    max_configs: int,
    seed: int,
    p_mutate: float = 0.60,
    p_swap_donch_period: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Neighborhood generator for Option 2 configs (entry_any = OR-of-AND clauses).
    - Keeps clause structure fixed
    - Perturbs thresholds within discrete neighborhoods
    - Optionally swaps Donchian 20/55 indicators
    - Lightly perturbs risk knobs (sl_param, leverage) in a bounded way
    """
    rng = random.Random(int(seed))

    if not isinstance(winner_cfg, dict):
        raise ValueError("winner_cfg must be a dict")
    if not winner_cfg.get("entry_any"):
        raise ValueError("winner_cfg does not contain entry_any")

    base = copy.deepcopy(winner_cfg)
    side = str(base.get("side", "long")).lower()
    entry_any = base.get("entry_any")
    if not isinstance(entry_any, list) or not entry_any:
        raise ValueError("winner_cfg.entry_any must be a non-empty list")

    # Risk neighborhood (minimal)
    risk = dict(base.get("risk") or {})
    base_sl_type = str(risk.get("sl_type", "PCT"))
    base_sl_param = float(risk.get("sl_param", 0.01) or 0.01)
    base_lev = float(risk.get("max_leverage", 2.0) or 2.0)

    sl_grid_pct = sorted(set([base_sl_param, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02]))
    lev_grid = sorted(set([base_lev, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0]))

    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    attempts = 0
    max_attempts = int(max(1000, max_configs * 50))

    while len(out) < int(max_configs) and attempts < max_attempts:
        attempts += 1
        cfg = copy.deepcopy(base)

        # Mutate entry_any thresholds
        ea2: List[List[Dict[str, Any]]] = []
        for clause in entry_any:
            if not isinstance(clause, list):
                continue
            new_clause: List[Dict[str, Any]] = []
            for cond in clause:
                if not isinstance(cond, dict):
                    continue
                c2 = dict(cond)

                ind = str(c2.get("indicator", ""))
                op = str(c2.get("operator", ""))
                thr = c2.get("threshold", 0.0)

                # Do not mutate the hard gate thresholds (keep 0.5 stable)
                if ind in ("trend_up_1h", "trend_down_1h"):
                    new_clause.append(c2)
                    continue

                # Optional donch period swap
                c2["indicator"] = _maybe_swap_donch_period(
                    ind, p_swap=float(p_swap_donch_period), rng=rng
                )

                # Threshold perturbation
                if rng.random() < float(p_mutate):
                    opts = _neighbor_thresholds(c2["indicator"], op, float(thr))
                    c2["threshold"] = float(rng.choice(opts))

                new_clause.append(c2)

            ea2.append(new_clause)

        cfg["entry_any"] = ea2

        # Minimal risk perturbation
        risk2 = dict(cfg.get("risk") or {})
        if base_sl_type.upper() == "PCT":
            if rng.random() < 0.50:
                risk2["sl_param"] = float(rng.choice(sl_grid_pct))
        if rng.random() < 0.35:
            risk2["max_leverage"] = float(rng.choice(lev_grid))
        cfg["risk"] = risk2

        # Keep name informative but do not affect dedupe key (dedupe ignores name)
        cfg["strategy_name"] = f"NEIGH_{side}_entryany"
        k = _cfg_key(cfg)
        if k in seen:
            continue
        seen.add(k)
        out.append(cfg)

    return out

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate JSONL grid for batch runner")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--max-configs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--entry-any",
        action="store_true",
        help="Generate Option 2 (DNF-lite) configs using entry_any (OR-of-AND)",
    )

    ap.add_argument(
        "--archetype",
        default="mixed",
        choices=["mixed", "pullback", "breakout", "squeeze_breakout"],
        help=(
            "Archetype family for entry_any generation. "
            "mixed = pullback cadence clause + breakout/squeeze clauses."
        ),
    )

    # Neighborhood injection
    ap.add_argument(
        "--winner-config",
        default=None,
        help="Path to winner config.json (from top/0001_.../config.json)",
    )
    ap.add_argument(
        "--add-neighborhood",
        action="store_true",
        help="Inject a neighborhood grid around the winner into the output",
    )
    ap.add_argument(
        "--neighborhood-n",
        type=int,
        default=2000,
        help="How many neighborhood configs to generate (before dedupe/cap).",
    )
    ap.add_argument(
        "--neighborhood-only",
        action="store_true",
        help="Output only the neighborhood grid (ignores broad space)",
    )
    ap.add_argument(
        "--neigh-ema50-toggle",
        action="store_true",
        help="Also test adding ema_50 trend filter (doubles neighborhood size)",
    )

    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    max_configs = int(args.max_configs)
    seed = int(args.seed)

    cfgs: List[Dict[str, Any]] = []

    # Neighborhood first (if requested)
    neighborhood: List[Dict[str, Any]] = []
    if args.add_neighborhood or args.neighborhood_only:
        if not args.winner_config:
            raise SystemExit("--winner-config is required with --add-neighborhood")
        winner = _load_winner_config(args.winner_config)
        # Auto-detect entry_any vs legacy
        if winner.get("entry_any"):
            neighborhood = build_neighborhood_from_winner_entry_any(
                winner,
                max_configs=int(args.neighborhood_n),
                seed=int(seed),
            )
        else:
            neighborhood = build_neighborhood_from_winner(
                winner,
                include_ema50_toggle=bool(args.neigh_ema50_toggle),
            )
        cfgs.extend(neighborhood)

    if not args.neighborhood_only:
        # Fill remaining budget with broad configs, excluding neighborhood dupes.
        exclude = {_cfg_key(c) for c in cfgs}
        remaining = max(0, max_configs - len(cfgs))

        if args.entry_any:
            sampled = build_space_entry_any(
                max_configs=remaining,
                seed=seed,
                exclude_keys=exclude,
                archetype=str(args.archetype),
            )
        else:
            space = build_space()
            sampled = sample_space(
                space,
                max_configs=remaining,
                seed=seed,
                exclude_keys=exclude,
            )

        cfgs.extend(sampled)

    cfgs = _dedupe_preserve_order(cfgs)

    if max_configs > 0:
        cfgs = cfgs[:max_configs]

    n = _jsonl_write(out_path, cfgs)
    print(f"Wrote {n} configs to: {out_path}")

    if neighborhood:
        print(f"Neighborhood injected: {len(neighborhood)} (before dedupe/cap)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())