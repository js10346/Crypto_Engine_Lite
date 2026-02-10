from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# =============================================================================
# Drift spec (v1)
# =============================================================================
# Single source of truth for what "drift" means. Both the generator and the UI
# should reference this, so the app never promises ranges that the generator
# doesn't actually use.
DCA_DRIFT_SPEC_V1: Dict[str, Any] = {
    "version": 1,
    "intensities": {
        # Neighborhood mode maps these to width.
        "conservative": {"width": "narrow"},
        "balanced": {"width": "medium"},
        "aggressive": {"width": "wide"},
    },
    "universes": {
        # Small, stable categorical universes
        "deposit_freqs": ["none", "weekly", "monthly"],
        "buy_freqs": ["weekly", "monthly"],
        "buy_filters": [
            "none",
            "below_ema",
            "rsi_below",
            "macd_bull",
            "bb_z_below",
            "adx_above",
            "donch_pos_below",
        ],
        "ema_lens": [20, 50, 100, 200],
        "rsi_thrs": [25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
        "bb_z_thrs": [-2.5, -2.0, -1.5, -1.0, -0.5],
        "adx_thrs": [15.0, 20.0, 25.0, 30.0],
        "donch_pos_thrs": [0.10, 0.20, 0.30, 0.40],
        # Numeric bins for neighborhood mode
        "money_bins": [0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0],
        "alloc_bins": [0.25, 0.50, 0.75, 0.90, 1.0],
        "sl_bins": [0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
        "tp_bins": [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00, 1.50, 2.00],
        "frac_bins": [0.0, 0.25, 0.50, 0.75, 1.0],
        "trail_bins": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
        "hold_bins": [0, 30, 60, 90, 180, 365, 730],
        # Random mode universes (vary by intensity)
        "random": {
            "conservative": {
                "deposit_amounts": [50.0, 100.0],
                "buy_amounts": [50.0, 100.0],
                "max_alloc_pcts": [0.75, 0.90, 1.0],
                "sl_pcts": [0.0, 0.05, 0.10, 0.15, 0.20],
                "tp_pcts": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
                "tp_sell_fracs": [0.50, 1.0],
                "reserve_fracs": [0.0, 0.25, 0.50],
                "trail_pcts": [0.0, 0.05, 0.10, 0.15],
                "max_hold_bars": [0, 30, 60, 90, 180, 365],
            },
            "balanced": {
                "deposit_amounts": [25.0, 50.0, 100.0, 200.0],
                "buy_amounts": [25.0, 50.0, 100.0, 200.0],
                "max_alloc_pcts": [0.50, 0.75, 0.90, 1.0],
                "sl_pcts": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                "tp_pcts": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
                "tp_sell_fracs": [0.25, 0.50, 1.0],
                "reserve_fracs": [0.0, 0.25, 0.50, 0.75, 1.0],
                "trail_pcts": [0.0, 0.05, 0.10, 0.15, 0.20],
                "max_hold_bars": [0, 30, 60, 90, 180, 365],
            },
            "aggressive": {
                "deposit_amounts": [10.0, 25.0, 50.0, 100.0, 200.0, 400.0],
                "buy_amounts": [10.0, 25.0, 50.0, 100.0, 200.0, 400.0],
                "max_alloc_pcts": [0.25, 0.50, 0.75, 0.90, 1.0],
                "sl_pcts": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
                "tp_pcts": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60],
                "tp_sell_fracs": [0.25, 0.50, 0.75, 1.0],
                "reserve_fracs": [0.0, 0.25, 0.50, 0.75, 1.0],
                "trail_pcts": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
                "max_hold_bars": [0, 30, 60, 90, 180, 365, 730],
            },
        },
    },
}

def _resolve_intensity(intensity: str) -> str:
    x = str(intensity or "balanced").strip().lower()
    return x if x in DCA_DRIFT_SPEC_V1["intensities"] else "balanced"

def _intensity_to_width(intensity: str, fallback_width: str = "medium") -> str:
    inten = _resolve_intensity(intensity)
    w = str(DCA_DRIFT_SPEC_V1["intensities"][inten].get("width") or fallback_width).strip().lower()
    return w if w in {"narrow", "medium", "wide"} else "medium"


# =============================================================================
# Utilities
# =============================================================================

def _canonical_json(obj: Any) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(_canonical_json(r))
            f.write("\n")
    return int(len(rows))


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _pick(rng: random.Random, xs: List[Any]) -> Any:
    return xs[rng.randrange(0, len(xs))]


def _cfg_key(cfg: Dict[str, Any]) -> str:
    # Stable dedupe key; strategy_name is constant, but keep it anyway.
    return _canonical_json(cfg)


def _read_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Base config must be a JSON object")
    return obj


def _as_float(x: Any, default: float) -> float:
    try:
        v = float(x)
        return v if v == v else float(default)
    except Exception:
        return float(default)


def _as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _as_str(x: Any, default: str) -> str:
    s = str(x) if x is not None else str(default)
    s = s.strip()
    return s if s else str(default)


def _neighbor_bins(baseline: float, bins: Sequence[float], *, width: str) -> List[float]:
    """
    Pick discrete neighbors from bins around baseline.
    width:
      - narrow: nearest 3
      - medium: nearest 5
      - wide: all bins
    Always includes baseline itself.
    """
    b = float(baseline)
    bins2 = [float(x) for x in bins]
    bins2 = [x for x in bins2 if x == x]  # drop NaN
    if not bins2:
        return [b]

    w = str(width).strip().lower()
    if w == "wide":
        out = list(bins2)
    else:
        k = 3 if w == "narrow" else 5
        out = sorted(bins2, key=lambda x: abs(x - b))[:k]

    out.append(b)
    return sorted({float(x) for x in out})


def _neighbor_enum(baseline: str, options: Sequence[str], *, width: str) -> List[str]:
    """
    Deterministic neighbor selection for enums.
      - narrow: baseline only
      - medium: baseline + next option in list (if exists)
      - wide: all options
    """
    opts = [str(x) for x in options]
    if not opts:
        return [str(baseline)]

    b = str(baseline).strip().lower()
    opts_l = [o.strip().lower() for o in opts]
    if b not in opts_l:
        b = opts_l[0]

    w = str(width).strip().lower()
    if w == "wide":
        return list(opts_l)
    if w == "narrow":
        return [b]

    # medium
    i = opts_l.index(b)
    nxt = opts_l[(i + 1) % len(opts_l)]
    if nxt == b:
        return [b]
    return [b, nxt]


# =============================================================================
# Entry logic helpers
# =============================================================================

def _cond(
    indicator: str,
    operator: str,
    *,
    threshold: Optional[float] = None,
    ref_indicator: Optional[str] = None,
) -> Dict[str, Any]:
    d: Dict[str, Any] = {"indicator": str(indicator), "operator": str(operator)}
    if ref_indicator is not None and str(ref_indicator).strip():
        d["ref_indicator"] = str(ref_indicator)
        d["threshold"] = float(threshold or 0.0)
    else:
        d["threshold"] = float(threshold or 0.0)
    return d


def _entry_logic_from_buy_filter(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backward-compatible conversion: existing buy_filter -> entry_logic.
    If buy_filter is "none", entry_logic will have empty clauses (always-true).
    """
    f = str(params.get("buy_filter", "none") or "none").strip().lower()

    if f in {"none", ""}:
        return {"regime": [], "clauses": []}

    if f == "below_ema":
        ema_len = int(params.get("ema_len", 200) or 200)
        return {"regime": [], "clauses": [[_cond("close", "<", ref_indicator=f"ema_{ema_len}")]]}

    if f == "rsi_below":
        thr = float(params.get("rsi_thr", 40.0) or 40.0)
        return {"regime": [], "clauses": [[_cond("rsi_14", "<=", threshold=thr)]]}

    if f == "macd_bull":
        thr = float(params.get("macd_hist_thr", 0.0) or 0.0)
        return {"regime": [], "clauses": [[_cond("macd_hist_12_26_9", ">=", threshold=thr)]]}

    if f == "bb_z_below":
        thr = float(params.get("bb_z_thr", -1.0) or -1.0)
        return {"regime": [], "clauses": [[_cond("bb_z_20", "<=", threshold=thr)]]}

    if f == "adx_above":
        thr = float(params.get("adx_thr", 20.0) or 20.0)
        return {"regime": [], "clauses": [[_cond("adx_14", ">=", threshold=thr)]]}

    if f == "donch_pos_below":
        thr = float(params.get("donch_pos_thr", 0.20) or 0.20)
        return {"regime": [], "clauses": [[_cond("donch_pos_20", "<=", threshold=thr)]]}

    return {"regime": [], "clauses": []}



def _mutate_entry_logic(
    rng: random.Random,
    base_logic: Dict[str, Any],
    *,
    width: str,
    ema_lens: Sequence[int],
    rsi_thrs: Sequence[float],
    bb_z_thrs: Sequence[float],
    adx_thrs: Sequence[float],
    donch_pos_thrs: Sequence[float],
) -> Dict[str, Any]:
    """Small, deterministic-ish mutations for neighborhood mode.
    We keep structure the same; only nudge thresholds / EMA lens.
    """
    try:
        logic = json.loads(_canonical_json(base_logic))
    except Exception:
        logic = {"regime": [], "clauses": []}

    w = _intensity_to_width(intensity, fallback_width=str(width))

    def mut_cond(c: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(c, dict):
            return c
        ind = str(c.get("indicator") or c.get("feature") or "").strip()
        ref = str(c.get("ref_indicator") or c.get("rhs") or "").strip()

        # EMA lens nudge (close vs ema_X)
        if ref.startswith("ema_"):
            try:
                base_len = int(ref.split("_", 1)[1])
            except Exception:
                base_len = 200
            choices = _neighbor_enum(str(ref), [f"ema_{int(x)}" for x in ema_lens], width=w)
            # _neighbor_enum returns strings; pick one and keep format
            ref2 = str(_pick(rng, list(choices)))
            c["ref_indicator"] = ref2

        # Numeric thresholds
        thr = c.get("threshold", c.get("value", None))
        if thr is None:
            return c
        thr0 = _as_float(thr, 0.0)

        if ind == "rsi_14":
            choices = _neighbor_bins(thr0, list(rsi_thrs), width=w)
            c["threshold"] = float(_pick(rng, list(choices)))
        elif ind == "bb_z_20":
            choices = _neighbor_bins(thr0, list(bb_z_thrs), width=w)
            c["threshold"] = float(_pick(rng, list(choices)))
        elif ind == "adx_14":
            choices = _neighbor_bins(thr0, list(adx_thrs), width=w)
            c["threshold"] = float(_pick(rng, list(choices)))
        elif ind == "donch_pos_20":
            choices = _neighbor_bins(thr0, list(donch_pos_thrs), width=w)
            c["threshold"] = float(_pick(rng, list(choices)))
        elif ind.startswith("macd_hist"):
            # Keep MACD hist threshold near baseline; allow small nudges around 0.0
            bins = [-0.5, -0.25, 0.0, 0.25, 0.5]
            choices = _neighbor_bins(thr0, bins, width=w)
            c["threshold"] = float(_pick(rng, list(choices)))

        return c

    # regime
    reg = logic.get("regime")
    if isinstance(reg, list):
        logic["regime"] = [mut_cond(c) for c in reg if isinstance(c, dict)]

    clauses = logic.get("clauses")
    if isinstance(clauses, list):
        new_clauses = []
        for clause in clauses:
            if isinstance(clause, list):
                new_clauses.append([mut_cond(c) for c in clause if isinstance(c, dict)])
        logic["clauses"] = new_clauses

    return logic


def _normalize_base_params(base: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts either:
      - full config: {"strategy_name","side","params":{...}}
      - params-only: {"deposit_freq":...,"buy_freq":...}
    Returns a params dict with defaults.
    """
    params_in = base.get("params") if isinstance(base.get("params"), dict) else base
    if not isinstance(params_in, dict):
        params_in = {}

    entry_logic_in = params_in.get("entry_logic")
    if not isinstance(entry_logic_in, dict):
        entry_logic_in = None

    # Defaults mirror strategies.dca_swing.DEFAULT_CONFIG
    p: Dict[str, Any] = {
        "deposit_freq": _as_str(params_in.get("deposit_freq"), "weekly").lower(),
        "deposit_amount_usd": _as_float(params_in.get("deposit_amount_usd"), 50.0),
        "buy_freq": _as_str(params_in.get("buy_freq"), "weekly").lower(),
        "buy_amount_usd": _as_float(params_in.get("buy_amount_usd"), 50.0),

        # Entry logic (preferred) + legacy fallback
        "entry_logic": entry_logic_in,
        "buy_filter": _as_str(params_in.get("buy_filter"), "none").lower(),
        "ema_len": _as_int(params_in.get("ema_len"), 200),
        "rsi_thr": _as_float(params_in.get("rsi_thr"), 40.0),
        "macd_hist_thr": _as_float(params_in.get("macd_hist_thr"), 0.0),
        "bb_z_thr": _as_float(params_in.get("bb_z_thr"), -1.0),
        "adx_thr": _as_float(params_in.get("adx_thr"), 20.0),
        "donch_pos_thr": _as_float(params_in.get("donch_pos_thr"), 0.20),

        "max_alloc_pct": _as_float(params_in.get("max_alloc_pct"), 1.0),

        "sl_pct": _as_float(params_in.get("sl_pct"), 0.0),
        "trail_pct": _as_float(params_in.get("trail_pct"), 0.0),
        "max_hold_bars": _as_int(params_in.get("max_hold_bars"), 0),

        "tp_pct": _as_float(params_in.get("tp_pct"), 0.0),
        "tp_sell_fraction": _as_float(params_in.get("tp_sell_fraction"), 1.0),
        "reserve_frac_of_proceeds": _as_float(params_in.get("reserve_frac_of_proceeds", params_in.get("reserve_frac")), 0.0),
    }

    # Basic cleanup
    if p["deposit_freq"] in {"none", "off", "0", ""}:
        p["deposit_freq"] = "none"
        p["deposit_amount_usd"] = 0.0

    allowed_filters = {
        "none",
        "below_ema",
        "rsi_below",
        "macd_bull",
        "bb_z_below",
        "adx_above",
        "donch_pos_below",
    }
    if p["buy_filter"] not in allowed_filters:
        p["buy_filter"] = "none"

    # If entry_logic is missing, derive from buy_filter
    if p["entry_logic"] is None:
        p["entry_logic"] = _entry_logic_from_buy_filter(p)

    return p


def _load_overrides(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Load drift override JSON (if provided). Schema is intentionally simple:

      {
        "<param_key>": {"min": <number>, "max": <number>},
        "<enum_key>": {"options": ["a","b", ...]}
      }

    Supported keys (v1):
      deposit_freq, buy_freq, buy_filter
      deposit_amount_usd, buy_amount_usd
      max_alloc_pct
      sl_pct, tp_pct, trail_pct
      tp_sell_fraction, reserve_frac_of_proceeds
      max_hold_bars
    """
    if not path:
        return None
    p = Path(str(path))
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text())
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _ov_minmax(ov: Optional[Dict[str, Any]], key: str) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(ov, dict):
        return (None, None)
    v = ov.get(key)
    if not isinstance(v, dict):
        return (None, None)
    mn = v.get("min")
    mx = v.get("max")
    try:
        mn_f = float(mn) if mn is not None else None
    except Exception:
        mn_f = None
    try:
        mx_f = float(mx) if mx is not None else None
    except Exception:
        mx_f = None
    if (mn_f is not None) and (mx_f is not None) and (mn_f > mx_f):
        mn_f, mx_f = mx_f, mn_f
    return (mn_f, mx_f)


def _ov_options(ov: Optional[Dict[str, Any]], key: str) -> Optional[List[str]]:
    if not isinstance(ov, dict):
        return None
    v = ov.get(key)
    if not isinstance(v, dict):
        return None
    opts = v.get("options")
    if not isinstance(opts, list):
        return None
    out: List[str] = []
    for x in opts:
        s = str(x).strip()
        if s:
            out.append(s)
    return out or None


def _filter_numeric(values: Sequence[float], mn: Optional[float], mx: Optional[float], *, include: Optional[Sequence[float]] = None) -> List[float]:
    xs = [float(x) for x in values]
    if mn is not None:
        xs = [x for x in xs if x >= float(mn) - 1e-12]
    if mx is not None:
        xs = [x for x in xs if x <= float(mx) + 1e-12]
    if include:
        for x in include:
            try:
                xs.append(float(x))
            except Exception:
                pass
    # de-dupe + sort
    xs2 = sorted({float(x) for x in xs})
    return xs2


def _filter_int(values: Sequence[int], mn: Optional[int], mx: Optional[int], *, include: Optional[Sequence[int]] = None) -> List[int]:
    xs = [int(x) for x in values]
    if mn is not None:
        xs = [x for x in xs if x >= int(mn)]
    if mx is not None:
        xs = [x for x in xs if x <= int(mx)]
    if include:
        for x in include:
            try:
                xs.append(int(x))
            except Exception:
                pass
    xs2 = sorted({int(x) for x in xs})
    return xs2

def build_dca_cfg(
    *,
    deposit_freq: str,
    deposit_amount_usd: float,
    buy_freq: str,
    buy_amount_usd: float,
    max_alloc_pct: float,
    sl_pct: float,
    tp_pct: float,
    tp_sell_fraction: float,
    reserve_frac_of_proceeds: float,
    # new exits
    trail_pct: float = 0.0,
    max_hold_bars: int = 0,
    # entry (either legacy or logic)
    buy_filter: str = "none",
    ema_len: Optional[int] = None,
    rsi_thr: Optional[float] = None,
    macd_hist_thr: Optional[float] = None,
    bb_z_thr: Optional[float] = None,
    adx_thr: Optional[float] = None,
    donch_pos_thr: Optional[float] = None,
    entry_logic: Optional[Dict[str, Any]] = None,
    is_baseline: bool = False,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "deposit_freq": str(deposit_freq),
        "deposit_amount_usd": float(deposit_amount_usd),
        "buy_freq": str(buy_freq),
        "buy_amount_usd": float(buy_amount_usd),

        "max_alloc_pct": float(_clamp01(max_alloc_pct)),

        "sl_pct": float(max(0.0, sl_pct)),
        "trail_pct": float(max(0.0, trail_pct)),
        "max_hold_bars": int(max(0, int(max_hold_bars or 0))),

        "tp_pct": float(max(0.0, tp_pct)),
        "tp_sell_fraction": float(_clamp01(tp_sell_fraction)),
        "reserve_frac_of_proceeds": float(_clamp01(reserve_frac_of_proceeds)),
    }

    # Keep legacy fields for readability/backward-compat, but always set entry_logic.
    params["buy_filter"] = str(buy_filter)

    if buy_filter == "below_ema":
        params["ema_len"] = int(ema_len or 200)
    if buy_filter == "rsi_below":
        params["rsi_thr"] = float(rsi_thr or 40.0)
    if buy_filter == "macd_bull":
        params["macd_hist_thr"] = float(macd_hist_thr or 0.0)
    if buy_filter == "bb_z_below":
        params["bb_z_thr"] = float(bb_z_thr or -1.0)
    if buy_filter == "adx_above":
        params["adx_thr"] = float(adx_thr or 20.0)
    if buy_filter == "donch_pos_below":
        params["donch_pos_thr"] = float(donch_pos_thr or 0.20)

    # If TP is disabled, make sell/reserve effectively no-op.
    if params["tp_pct"] <= 0.0:
        params["tp_sell_fraction"] = 1.0
        params["reserve_frac_of_proceeds"] = 0.0

    # If there is no selling, reserve is pointless.
    if params["tp_sell_fraction"] <= 0.0:
        params["reserve_frac_of_proceeds"] = 0.0

    if isinstance(entry_logic, dict):
        params["entry_logic"] = entry_logic
    else:
        params["entry_logic"] = _entry_logic_from_buy_filter(params)

    if bool(is_baseline):
        params["__baseline__"] = True

    return {
        "strategy_name": "dca_swing",
        "side": "long",
        "params": params,
    }


# =============================================================================
# Generators
# =============================================================================

def _random_entry_logic(
    rng: random.Random,
    *,
    ema_lens: List[int],
    rsi_thrs: List[float],
    bb_z_thrs: List[float],
    adx_thrs: List[float],
    donch_pos_thrs: List[float],
) -> Dict[str, Any]:
    """
    A small, v1 “logic builder” search space: regime gate + OR-of-AND triggers.
    This intentionally stays simple and auditable.
    """
    # regime choices
    regime_mode = _pick(rng, ["none", "uptrend", "strong", "uptrend_strong", "range"])
    regime: List[Dict[str, Any]] = []
    if regime_mode in {"uptrend", "uptrend_strong"}:
        regime.append(_cond("close", ">", ref_indicator="ema_200"))
    if regime_mode in {"strong", "uptrend_strong"}:
        thr = float(_pick(rng, adx_thrs))
        regime.append(_cond("adx_14", ">=", threshold=thr))
    if regime_mode == "range":
        thr = float(_pick(rng, adx_thrs))
        regime.append(_cond("adx_14", "<=", threshold=thr))

    # trigger packages (each becomes a clause)
    ema_len = int(_pick(rng, ema_lens))
    rsi_thr = float(_pick(rng, rsi_thrs))
    bb_z_thr = float(_pick(rng, bb_z_thrs))
    donch_thr = float(_pick(rng, donch_pos_thrs))

    packages: List[str] = [
        "dip",
        "rsi",
        "bb",
        "macd",
        "donch",
        "dip+rsi",
        "dip+bb",
        "donch+rsi",
    ]
    k = int(_pick(rng, [1, 1, 2, 2, 3]))  # bias toward 1–2
    chosen = rng.sample(packages, k=k)

    clauses: List[List[Dict[str, Any]]] = []
    for name in chosen:
        if name == "dip":
            clauses.append([_cond("close", "<", ref_indicator=f"ema_{ema_len}")])
        elif name == "rsi":
            clauses.append([_cond("rsi_14", "<=", threshold=rsi_thr)])
        elif name == "bb":
            clauses.append([_cond("bb_z_20", "<=", threshold=bb_z_thr)])
        elif name == "macd":
            clauses.append([_cond("macd_hist_12_26_9", ">=", threshold=0.0)])
        elif name == "donch":
            clauses.append([_cond("donch_pos_20", "<=", threshold=donch_thr)])
        elif name == "dip+rsi":
            clauses.append([
                _cond("close", "<", ref_indicator=f"ema_{ema_len}"),
                _cond("rsi_14", "<=", threshold=rsi_thr),
            ])
        elif name == "dip+bb":
            clauses.append([
                _cond("close", "<", ref_indicator=f"ema_{ema_len}"),
                _cond("bb_z_20", "<=", threshold=bb_z_thr),
            ])
        elif name == "donch+rsi":
            clauses.append([
                _cond("donch_pos_20", "<=", threshold=donch_thr),
                _cond("rsi_14", "<=", threshold=rsi_thr),
            ])

    return {"regime": regime, "clauses": clauses}


def generate_grid_random(
    *,
    n: int,
    seed: int,
    logic_frac: float = 0.35,
    intensity: str = "balanced",
    base_cfg: Optional[Dict[str, Any]] = None,
    vary_groups: Optional[Set[str]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    rng = random.Random(int(seed))
    inten = _resolve_intensity(intensity)

    uni = DCA_DRIFT_SPEC_V1["universes"]

    deposit_freqs = list(uni["deposit_freqs"])
    buy_freqs = list(uni["buy_freqs"])

    # Random universes vary by intensity.
    r_uni = dict((uni.get("random") or {}).get(inten) or (uni.get("random") or {}).get("balanced") or {})

    deposit_amounts = list(r_uni.get("deposit_amounts") or [25.0, 50.0, 100.0, 200.0])
    buy_amounts = list(r_uni.get("buy_amounts") or [25.0, 50.0, 100.0, 200.0])

    # Legacy single-filter options (kept) + v1 logic builder mode.
    buy_filters = list(uni["buy_filters"])
    ema_lens = list(uni["ema_lens"])
    rsi_thrs = list(uni["rsi_thrs"][:-1])  # keep random a bit tighter than neighborhood
    bb_z_thrs = list(uni["bb_z_thrs"])
    adx_thrs = list(uni["adx_thrs"])
    donch_pos_thrs = list(uni["donch_pos_thrs"])

    max_alloc_pcts = list(r_uni.get("max_alloc_pcts") or [0.5, 0.75, 0.9, 1.0])

    sl_pcts = list(r_uni.get("sl_pcts") or [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    tp_pcts = list(r_uni.get("tp_pcts") or [0.0, 0.05, 0.10, 0.15, 0.20, 0.30])
    tp_sell_fracs = list(r_uni.get("tp_sell_fracs") or [0.25, 0.50, 1.0])
    reserve_fracs = list(r_uni.get("reserve_fracs") or [0.0, 0.25, 0.50, 0.75, 1.0])

    # New exits
    trail_pcts = list(r_uni.get("trail_pcts") or [0.0, 0.05, 0.10, 0.15, 0.20])
    max_hold_bars = list(r_uni.get("max_hold_bars") or [0, 30, 60, 90, 180, 365])

    bp = _normalize_base_params(base_cfg) if isinstance(base_cfg, dict) else None
    vg = {str(x).strip().lower() for x in (vary_groups or set()) if str(x).strip()}
    if not vg:
        vg = {"deposits", "buys", "filter", "logic", "alloc", "risk"}


    ov = overrides if isinstance(overrides, dict) else None

    # Override categorical universes (subsets)
    dep_opts = _ov_options(ov, "deposit_freq")
    if dep_opts is not None:
        deposit_freqs = [x for x in deposit_freqs if str(x) in set(dep_opts)]
        if not deposit_freqs:
            deposit_freqs = list(uni["deposit_freqs"])

    buyf_opts = _ov_options(ov, "buy_freq")
    if buyf_opts is not None:
        buy_freqs = [x for x in buy_freqs if str(x) in set(buyf_opts)]
        if not buy_freqs:
            buy_freqs = list(uni["buy_freqs"])

    filt_opts = _ov_options(ov, "buy_filter")
    if filt_opts is not None:
        buy_filters = [x for x in buy_filters if str(x) in set(filt_opts)]
        if not buy_filters:
            buy_filters = list(uni["buy_filters"])

    # Override numeric universes (min/max)
    mn, mx = _ov_minmax(ov, "deposit_amount_usd")
    deposit_amounts = _filter_numeric(
        deposit_amounts,
        mn,
        mx,
        include=[float(bp["deposit_amount_usd"])] if isinstance(bp, dict) else None,
    ) or deposit_amounts

    mn, mx = _ov_minmax(ov, "buy_amount_usd")
    buy_amounts = _filter_numeric(
        buy_amounts,
        mn,
        mx,
        include=[float(bp["buy_amount_usd"])] if isinstance(bp, dict) else None,
    ) or buy_amounts

    mn, mx = _ov_minmax(ov, "max_alloc_pct")
    max_alloc_pcts = _filter_numeric(
        max_alloc_pcts,
        mn,
        mx,
        include=[float(bp["max_alloc_pct"])] if isinstance(bp, dict) else None,
    ) or max_alloc_pcts

    mn, mx = _ov_minmax(ov, "sl_pct")
    sl_pcts = _filter_numeric(sl_pcts, mn, mx, include=[float(bp["sl_pct"])] if isinstance(bp, dict) else None) or sl_pcts

    mn, mx = _ov_minmax(ov, "tp_pct")
    tp_pcts = _filter_numeric(tp_pcts, mn, mx, include=[float(bp["tp_pct"])] if isinstance(bp, dict) else None) or tp_pcts

    mn, mx = _ov_minmax(ov, "trail_pct")
    trail_pcts = _filter_numeric(trail_pcts, mn, mx, include=[float(bp["trail_pct"])] if isinstance(bp, dict) else None) or trail_pcts

    mn, mx = _ov_minmax(ov, "tp_sell_fraction")
    tp_sell_fracs = _filter_numeric(tp_sell_fracs, mn, mx, include=[float(bp["tp_sell_fraction"])] if isinstance(bp, dict) else None) or tp_sell_fracs

    mn, mx = _ov_minmax(ov, "reserve_frac_of_proceeds")
    reserve_fracs = _filter_numeric(reserve_fracs, mn, mx, include=[float(bp["reserve_frac_of_proceeds"])] if isinstance(bp, dict) else None) or reserve_fracs

    mn, mx = _ov_minmax(ov, "max_hold_bars")
    mn_i = int(mn) if mn is not None else None
    mx_i = int(mx) if mx is not None else None
    max_hold_bars = _filter_int(
        max_hold_bars,
        mn_i,
        mx_i,
        include=[int(bp["max_hold_bars"])] if isinstance(bp, dict) else None,
    ) or max_hold_bars

    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    attempts = 0
    max_attempts = max(5000, int(n) * 60)

    while len(out) < int(n) and attempts < max_attempts:
        attempts += 1

        # deposits
        if (bp is not None) and ("deposits" not in vg):
            deposit_freq = str(bp["deposit_freq"])
            deposit_amount = float(bp["deposit_amount_usd"])
        else:
            deposit_freq = str(_pick(rng, deposit_freqs))
            deposit_amount = 0.0 if deposit_freq == "none" else float(_pick(rng, deposit_amounts))
        if str(deposit_freq).lower() == "none":
            deposit_amount = 0.0

        # buys
        if (bp is not None) and ("buys" not in vg):
            buy_freq = str(bp["buy_freq"])
            buy_amount = float(bp["buy_amount_usd"])
        else:
            buy_freq = str(_pick(rng, buy_freqs))
            buy_amount = float(_pick(rng, buy_amounts))

        # allocation
        if (bp is not None) and ("alloc" not in vg):
            max_alloc_pct = float(bp["max_alloc_pct"])
        else:
            max_alloc_pct = float(_pick(rng, max_alloc_pcts))

        # risk/exits
        if (bp is not None) and ("risk" not in vg):
            sl_pct = float(bp["sl_pct"])
            tp_pct = float(bp["tp_pct"])
            tp_sell_fraction = float(bp["tp_sell_fraction"])
            reserve_frac = float(bp["reserve_frac_of_proceeds"])
            trail_pct = float(bp["trail_pct"])
            hold_bars = int(bp["max_hold_bars"])
        else:
            sl_pct = float(_pick(rng, sl_pcts))
            tp_pct = float(_pick(rng, tp_pcts))
            tp_sell_fraction = float(_pick(rng, tp_sell_fracs))
            reserve_frac = float(_pick(rng, reserve_fracs))
            trail_pct = float(_pick(rng, trail_pcts))
            hold_bars = int(_pick(rng, max_hold_bars))

        # entry gate
        base_is_logic = bool(
            bp is not None
            and isinstance(bp.get("entry_logic"), dict)
            and isinstance(bp["entry_logic"].get("clauses"), list)
            and len(bp["entry_logic"].get("clauses") or []) > 0
            and str(bp.get("buy_filter", "none")).lower() in {"none", ""}
        )

        allow_logic = ("logic" in vg)
        allow_filter = ("filter" in vg)

        if (bp is not None) and (not allow_logic) and (not allow_filter):
            # Fully pinned to baseline entry mode.
            if base_is_logic:
                buy_filter = "none"
                ema_len = None
                rsi_thr = None
                macd_hist_thr = None
                bb_z_thr = None
                adx_thr = None
                donch_pos_thr = None
                entry_logic = dict(bp.get("entry_logic") or {"regime": [], "clauses": []})
            else:
                buy_filter = str(bp.get("buy_filter", "none"))
                ema_len = int(bp.get("ema_len", 200)) if buy_filter == "below_ema" else None
                rsi_thr = float(bp.get("rsi_thr", 40.0)) if buy_filter == "rsi_below" else None
                macd_hist_thr = float(bp.get("macd_hist_thr", 0.0)) if buy_filter == "macd_bull" else 0.0
                bb_z_thr = float(bp.get("bb_z_thr", -1.0)) if buy_filter == "bb_z_below" else None
                adx_thr = float(bp.get("adx_thr", 20.0)) if buy_filter == "adx_above" else None
                donch_pos_thr = float(bp.get("donch_pos_thr", 0.20)) if buy_filter == "donch_pos_below" else None
                entry_logic = None
        else:
            # Decide whether to emit logic mode or legacy filter mode.
            if allow_logic and allow_filter:
                use_logic = (rng.random() < float(logic_frac))
            elif allow_logic and (not allow_filter):
                use_logic = True
            else:
                use_logic = False

            if use_logic:
                buy_filter = "none"
                ema_len = None
                rsi_thr = None
                macd_hist_thr = None
                bb_z_thr = None
                adx_thr = None
                donch_pos_thr = None
                entry_logic = _random_entry_logic(
                    rng,
                    ema_lens=ema_lens,
                    rsi_thrs=rsi_thrs,
                    bb_z_thrs=bb_z_thrs,
                    adx_thrs=adx_thrs,
                    donch_pos_thrs=donch_pos_thrs,
                )
            else:
                buy_filter = str(_pick(rng, buy_filters))
                ema_len = int(_pick(rng, ema_lens)) if buy_filter == "below_ema" else None
                rsi_thr = float(_pick(rng, rsi_thrs)) if buy_filter == "rsi_below" else None
                macd_hist_thr = 0.0
                bb_z_thr = float(_pick(rng, bb_z_thrs)) if buy_filter == "bb_z_below" else None
                adx_thr = float(_pick(rng, adx_thrs)) if buy_filter == "adx_above" else None
                donch_pos_thr = float(_pick(rng, donch_pos_thrs)) if buy_filter == "donch_pos_below" else None
                entry_logic = None  # derived in build_dca_cfg

        cfg = build_dca_cfg(
            deposit_freq=deposit_freq,
            deposit_amount_usd=deposit_amount,
            buy_freq=buy_freq,
            buy_amount_usd=buy_amount,

            buy_filter=buy_filter,
            ema_len=ema_len,
            rsi_thr=rsi_thr,
            macd_hist_thr=macd_hist_thr,
            bb_z_thr=bb_z_thr,
            adx_thr=adx_thr,
            donch_pos_thr=donch_pos_thr,
            entry_logic=entry_logic,

            max_alloc_pct=max_alloc_pct,

            sl_pct=sl_pct,
            trail_pct=trail_pct,
            max_hold_bars=hold_bars,

            tp_pct=tp_pct,
            tp_sell_fraction=tp_sell_fraction,
            reserve_frac_of_proceeds=reserve_frac,
        )

        k = _cfg_key(cfg)
        if k in seen:
            continue
        seen.add(k)
        out.append(cfg)

    return out


def generate_grid_neighborhood(
    *,
    n: int,
    seed: int,
    base_cfg: Dict[str, Any],
    width: str = "medium",
    intensity: str = "balanced",
    vary_groups: Optional[Set[str]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Neighborhood generator around a baseline config.

    - Always includes the baseline config as the first row.
    - Samples discrete neighbor bins based on width.
    - vary_groups controls which groups are allowed to vary:
        {"deposits","buys","filter","logic","alloc","risk"}
      If None: all groups vary.
    """
    rng = random.Random(int(seed))
    w = str(width).strip().lower()
    if w not in {"narrow", "medium", "wide"}:
        w = "medium"

    if vary_groups is None:
        vary_groups = {"deposits", "buys", "filter", "logic", "alloc", "risk"}
    else:
        vary_groups = {str(x).strip().lower() for x in vary_groups if str(x).strip()}

    bp = _normalize_base_params(base_cfg)

    base_row = build_dca_cfg(
        deposit_freq=str(bp["deposit_freq"]),
        deposit_amount_usd=float(bp["deposit_amount_usd"]),
        buy_freq=str(bp["buy_freq"]),
        buy_amount_usd=float(bp["buy_amount_usd"]),

        buy_filter=str(bp["buy_filter"]),
        ema_len=int(bp.get("ema_len", 200)),
        rsi_thr=float(bp.get("rsi_thr", 40.0)),
        macd_hist_thr=float(bp.get("macd_hist_thr", 0.0)),
        bb_z_thr=float(bp.get("bb_z_thr", -1.0)),
        adx_thr=float(bp.get("adx_thr", 20.0)),
        donch_pos_thr=float(bp.get("donch_pos_thr", 0.20)),
        entry_logic=bp.get("entry_logic"),

        max_alloc_pct=float(bp["max_alloc_pct"]),

        sl_pct=float(bp["sl_pct"]),
        trail_pct=float(bp["trail_pct"]),
        max_hold_bars=int(bp["max_hold_bars"]),

        tp_pct=float(bp["tp_pct"]),
        tp_sell_fraction=float(bp["tp_sell_fraction"]),
        reserve_frac_of_proceeds=float(bp["reserve_frac_of_proceeds"]),
        is_baseline=True,
    )

    base_params = dict(base_row.get("params") or {})

    # Discrete universes (single source of truth)
    uni = DCA_DRIFT_SPEC_V1["universes"]

    deposit_freqs = list(uni["deposit_freqs"])
    buy_freqs = list(uni["buy_freqs"])
    buy_filters = list(uni["buy_filters"])

    money_bins = list(uni["money_bins"])
    alloc_bins = list(uni["alloc_bins"])
    sl_bins = list(uni["sl_bins"])
    tp_bins = list(uni["tp_bins"])
    frac_bins = list(uni["frac_bins"])
    ema_lens = list(uni["ema_lens"])
    rsi_thrs = list(uni["rsi_thrs"])
    bb_z_thrs = list(uni["bb_z_thrs"])
    adx_thrs = list(uni["adx_thrs"])
    donch_pos_thrs = list(uni["donch_pos_thrs"])

    trail_bins = list(uni["trail_bins"])
    hold_bins = list(uni["hold_bins"])
    ov = overrides if isinstance(overrides, dict) else None

    # Apply option subset overrides
    dep_opts = _ov_options(ov, "deposit_freq")
    if dep_opts is not None:
        deposit_freqs = [x for x in deposit_freqs if str(x) in set(dep_opts)] or deposit_freqs

    buyf_opts = _ov_options(ov, "buy_freq")
    if buyf_opts is not None:
        buy_freqs = [x for x in buy_freqs if str(x) in set(buyf_opts)] or buy_freqs

    filt_opts = _ov_options(ov, "buy_filter")
    if filt_opts is not None:
        buy_filters = [x for x in buy_filters if str(x) in set(filt_opts)] or buy_filters

    # Build per-field bins with numeric overrides
    dep_mn, dep_mx = _ov_minmax(ov, "deposit_amount_usd")
    buy_mn, buy_mx = _ov_minmax(ov, "buy_amount_usd")
    alloc_mn, alloc_mx = _ov_minmax(ov, "max_alloc_pct")
    sl_mn, sl_mx = _ov_minmax(ov, "sl_pct")
    tp_mn, tp_mx = _ov_minmax(ov, "tp_pct")
    sell_mn, sell_mx = _ov_minmax(ov, "tp_sell_fraction")
    res_mn, res_mx = _ov_minmax(ov, "reserve_frac_of_proceeds")
    trail_mn, trail_mx = _ov_minmax(ov, "trail_pct")
    hold_mn, hold_mx = _ov_minmax(ov, "max_hold_bars")
    hold_mn_i = int(hold_mn) if hold_mn is not None else None
    hold_mx_i = int(hold_mx) if hold_mx is not None else None

    money_bins_dep = _filter_numeric(money_bins, dep_mn, dep_mx, include=[float(base_params.get("deposit_amount_usd", 0.0) or 0.0)])
    money_bins_buy = _filter_numeric(money_bins, buy_mn, buy_mx, include=[float(base_params.get("buy_amount_usd", 0.0) or 0.0)])
    alloc_bins2 = _filter_numeric(alloc_bins, alloc_mn, alloc_mx, include=[float(base_params.get("max_alloc_pct", 1.0) or 1.0)]) or alloc_bins
    sl_bins2 = _filter_numeric(sl_bins, sl_mn, sl_mx, include=[float(base_params.get("sl_pct", 0.0) or 0.0)]) or sl_bins
    tp_bins2 = _filter_numeric(tp_bins, tp_mn, tp_mx, include=[float(base_params.get("tp_pct", 0.0) or 0.0)]) or tp_bins
    frac_bins_sell = _filter_numeric(frac_bins, sell_mn, sell_mx, include=[float(base_params.get("tp_sell_fraction", 1.0) or 1.0)]) or frac_bins
    frac_bins_res = _filter_numeric(frac_bins, res_mn, res_mx, include=[float(base_params.get("reserve_frac_of_proceeds", 0.0) or 0.0)]) or frac_bins
    trail_bins2 = _filter_numeric(trail_bins, trail_mn, trail_mx, include=[float(base_params.get("trail_pct", 0.0) or 0.0)]) or trail_bins
    hold_bins2 = _filter_int(hold_bins, hold_mn_i, hold_mx_i, include=[int(base_params.get("max_hold_bars", 0) or 0)]) or hold_bins


    def fixed_or(var: Any, values: List[Any], group: str) -> List[Any]:
        return values if (group in vary_groups) else [var]

    # deposits / buys
    dep_freq_choices = fixed_or(
        str(base_params.get("deposit_freq", "weekly")).lower(),
        _neighbor_enum(str(base_params.get("deposit_freq", "weekly")), deposit_freqs, width=w),
        "deposits",
    )
    dep_amt_choices = fixed_or(
        float(base_params.get("deposit_amount_usd", 0.0) or 0.0),
        _neighbor_bins(float(base_params.get("deposit_amount_usd", 0.0) or 0.0), money_bins_dep, width=w),
        "deposits",
    )
    buy_freq_choices = fixed_or(
        str(base_params.get("buy_freq", "weekly")).lower(),
        _neighbor_enum(str(base_params.get("buy_freq", "weekly")), buy_freqs, width=w),
        "buys",
    )
    buy_amt_choices = fixed_or(
        float(base_params.get("buy_amount_usd", 0.0) or 0.0),
        _neighbor_bins(float(base_params.get("buy_amount_usd", 0.0) or 0.0), money_bins_buy, width=w),
        "buys",
    )

    # entry / filter
    base_entry_logic = base_params.get("entry_logic") if isinstance(base_params.get("entry_logic"), dict) else None
    base_is_logic = bool(base_entry_logic and isinstance(base_entry_logic.get("clauses"), list) and len(base_entry_logic.get("clauses")) > 0) and (str(base_params.get("buy_filter", "none")).lower() in {"none", ""})

    if base_is_logic:
        # Logic-builder neighborhood: keep buy_filter pinned; nudge entry_logic instead.
        filt_choices = ["none"]
        ema_choices = [200]  # legacy knobs unused in logic mode
        rsi_choices = [40.0]
        bbz_choices = [-1.0]
        adx_choices = [20.0]
        donch_choices = [0.20]
    else:
        # Legacy single-filter neighborhood
        filt_choices = fixed_or(
            str(base_params.get("buy_filter", "none")).lower(),
            _neighbor_enum(str(base_params.get("buy_filter", "none")), buy_filters, width=w),
            "filter",
        )
        ema_choices = fixed_or(
            int(base_params.get("ema_len", 200) or 200),
            [int(x) for x in ema_lens],
            "filter",
        )
        rsi_choices = fixed_or(
            float(base_params.get("rsi_thr", 40.0) or 40.0),
            [float(x) for x in rsi_thrs],
            "filter",
        )
        bbz_choices = fixed_or(
            float(base_params.get("bb_z_thr", -1.0) or -1.0),
            [float(x) for x in bb_z_thrs],
            "filter",
        )
        adx_choices = fixed_or(
            float(base_params.get("adx_thr", 20.0) or 20.0),
            [float(x) for x in adx_thrs],
            "filter",
        )
        donch_choices = fixed_or(
            float(base_params.get("donch_pos_thr", 0.20) or 0.20),
            [float(x) for x in donch_pos_thrs],
            "filter",
        )

    # alloc / risk
    alloc_choices = fixed_or(
        float(base_params.get("max_alloc_pct", 1.0) or 1.0),
        _neighbor_bins(float(base_params.get("max_alloc_pct", 1.0) or 1.0), alloc_bins2, width=w),
        "alloc",
    )
    sl_choices = fixed_or(
        float(base_params.get("sl_pct", 0.0) or 0.0),
        _neighbor_bins(float(base_params.get("sl_pct", 0.0) or 0.0), sl_bins2, width=w),
        "risk",
    )
    tp_choices = fixed_or(
        float(base_params.get("tp_pct", 0.0) or 0.0),
        _neighbor_bins(float(base_params.get("tp_pct", 0.0) or 0.0), tp_bins2, width=w),
        "risk",
    )
    sell_choices = fixed_or(
        float(base_params.get("tp_sell_fraction", 1.0) or 1.0),
        _neighbor_bins(float(base_params.get("tp_sell_fraction", 1.0) or 1.0), frac_bins_sell, width=w),
        "risk",
    )
    reserve_choices = fixed_or(
        float(base_params.get("reserve_frac_of_proceeds", 0.0) or 0.0),
        _neighbor_bins(float(base_params.get("reserve_frac_of_proceeds", 0.0) or 0.0), frac_bins_res, width=w),
        "risk",
    )
    trail_choices = fixed_or(
        float(base_params.get("trail_pct", 0.0) or 0.0),
        _neighbor_bins(float(base_params.get("trail_pct", 0.0) or 0.0), trail_bins2, width=w),
        "risk",
    )
    hold_choices = fixed_or(
        int(base_params.get("max_hold_bars", 0) or 0),
        [int(x) for x in hold_bins2],
        "risk",
    )

    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    out.append(base_row)
    seen.add(_cfg_key(base_row))

    attempts = 0
    max_attempts = max(5000, int(n) * 80)

    while len(out) < int(n) and attempts < max_attempts:
        attempts += 1

        deposit_freq = str(_pick(rng, list(dep_freq_choices))).lower()
        deposit_amount = float(_pick(rng, list(dep_amt_choices)))
        if deposit_freq == "none":
            deposit_amount = 0.0

        buy_freq = str(_pick(rng, list(buy_freq_choices))).lower()
        buy_amount = float(_pick(rng, list(buy_amt_choices)))

        buy_filter = str(_pick(rng, list(filt_choices))).lower()
        ema_len = int(_pick(rng, list(ema_choices)))
        rsi_thr = float(_pick(rng, list(rsi_choices)))
        bb_z_thr = float(_pick(rng, list(bbz_choices)))
        adx_thr = float(_pick(rng, list(adx_choices)))
        donch_thr = float(_pick(rng, list(donch_choices)))

        max_alloc_pct = float(_pick(rng, list(alloc_choices)))

        sl_pct = float(_pick(rng, list(sl_choices)))
        tp_pct = float(_pick(rng, list(tp_choices)))
        tp_sell_fraction = float(_pick(rng, list(sell_choices)))
        reserve_frac = float(_pick(rng, list(reserve_choices)))

        trail_pct = float(_pick(rng, list(trail_choices)))
        hold_bars = int(_pick(rng, list(hold_choices)))


        entry_logic = None
        if base_is_logic:
            base_logic = base_entry_logic if isinstance(base_entry_logic, dict) else {"regime": [], "clauses": []}
            if "logic" in vary_groups:
                entry_logic = _mutate_entry_logic(
                    rng,
                    base_logic,
                    width=w,
                    ema_lens=ema_lens,
                    rsi_thrs=rsi_thrs,
                    bb_z_thrs=bb_z_thrs,
                    adx_thrs=adx_thrs,
                    donch_pos_thrs=donch_pos_thrs,
                )
            else:
                entry_logic = base_logic

        cfg = build_dca_cfg(
            deposit_freq=deposit_freq,
            deposit_amount_usd=deposit_amount,
            buy_freq=buy_freq,
            buy_amount_usd=buy_amount,
            buy_filter="none" if base_is_logic else buy_filter,
            ema_len=None if base_is_logic else (ema_len if buy_filter == "below_ema" else None),
            rsi_thr=None if base_is_logic else (rsi_thr if buy_filter == "rsi_below" else None),
            bb_z_thr=None if base_is_logic else (bb_z_thr if buy_filter == "bb_z_below" else None),
            adx_thr=None if base_is_logic else (adx_thr if buy_filter == "adx_above" else None),
            donch_pos_thr=None if base_is_logic else (donch_thr if buy_filter == "donch_pos_below" else None),
            entry_logic=entry_logic,

            max_alloc_pct=max_alloc_pct,

            sl_pct=sl_pct,
            trail_pct=trail_pct,
            max_hold_bars=hold_bars,

            tp_pct=tp_pct,
            tp_sell_fraction=tp_sell_fraction,
            reserve_frac_of_proceeds=reserve_frac,
        )

        k = _cfg_key(cfg)
        if k in seen:
            continue
        seen.add(k)
        out.append(cfg)

    return out



def preview_drift_table(
    *,
    mode: str,
    intensity: str = "balanced",
    base_cfg: Optional[Dict[str, Any]] = None,
    vary_groups: Optional[Set[str]] = None,
    logic_frac: float = 0.35,
    width: str = "medium",
    overrides: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """
    Returns a human-readable preview of what can drift (and how far) given the current
    generator settings. Designed to be used by the Streamlit UI.

    Columns: Dimension | Baseline | Drift
    """
    uni = DCA_DRIFT_SPEC_V1["universes"]
    inten = _resolve_intensity(intensity)
    vg = {str(x).strip().lower() for x in (vary_groups or set()) if str(x).strip()}
    if not vg:
        vg = {"deposits", "buys", "filter", "logic", "alloc", "risk"}

    bp = _normalize_base_params(base_cfg) if isinstance(base_cfg, dict) else None
    ov = overrides if isinstance(overrides, dict) else None
    w = _intensity_to_width(inten, fallback_width=str(width))

    # Local universes (apply overrides so preview matches generation)
    dep_freqs_u = _ov_options(ov, "deposit_freq") or list(uni["deposit_freqs"])
    buy_freqs_u = _ov_options(ov, "buy_freq") or list(uni["buy_freqs"])
    buy_filters_u = _ov_options(ov, "buy_filter") or list(uni["buy_filters"])

    dep_mn, dep_mx = _ov_minmax(ov, "deposit_amount_usd")
    buy_mn, buy_mx = _ov_minmax(ov, "buy_amount_usd")
    alloc_mn, alloc_mx = _ov_minmax(ov, "max_alloc_pct")
    sl_mn, sl_mx = _ov_minmax(ov, "sl_pct")
    tp_mn, tp_mx = _ov_minmax(ov, "tp_pct")
    sell_mn, sell_mx = _ov_minmax(ov, "tp_sell_fraction")
    res_mn, res_mx = _ov_minmax(ov, "reserve_frac_of_proceeds")
    trail_mn, trail_mx = _ov_minmax(ov, "trail_pct")
    hold_mn, hold_mx = _ov_minmax(ov, "max_hold_bars")
    hold_mn_i = int(hold_mn) if hold_mn is not None else None
    hold_mx_i = int(hold_mx) if hold_mx is not None else None

    def fmt_enum(xs: Sequence[Any], limit: int = 5) -> str:
        xs2 = [str(x) for x in xs]
        if len(xs2) <= limit:
            return ", ".join(xs2)
        return ", ".join(xs2[:limit]) + ", …"

    def fmt_range(xs: Sequence[float]) -> str:
        xs2 = [float(x) for x in xs]
        if not xs2:
            return ""
        lo = min(xs2)
        hi = max(xs2)
        if abs(lo - hi) < 1e-12:
            return f"{lo:g}"
        return f"{lo:g}–{hi:g}"

    def baseline_or(default: Any, key: str) -> Any:
        return (bp.get(key) if isinstance(bp, dict) else default)

    def base_is_logic() -> bool:
        if not isinstance(bp, dict):
            return False
        el = bp.get("entry_logic")
        return bool(
            isinstance(el, dict)
            and isinstance(el.get("clauses"), list)
            and len(el.get("clauses") or []) > 0
            and str(bp.get("buy_filter", "none")).lower() in {"none", ""}
        )

    rows: List[Dict[str, str]] = []

    # -------------------------
    # Deposits (funding)
    # -------------------------
    dep_freq0 = str(baseline_or("weekly", "deposit_freq")).lower()
    dep_amt0 = float(baseline_or(50.0, "deposit_amount_usd") or 0.0)

    if mode == "neighborhood":
        dep_freq_choices = _neighbor_enum(dep_freq0, list(dep_freqs_u), width=w) if "deposits" in vg else [dep_freq0]
        dep_amt_choices = _neighbor_bins(dep_amt0, _filter_numeric(list(uni["money_bins"]), dep_mn, dep_mx, include=[dep_amt0]), width=w) if "deposits" in vg else [dep_amt0]
    else:
        dep_freq_choices = list(dep_freqs_u) if "deposits" in vg else [dep_freq0]
        dep_amt_choices = _filter_numeric(list((uni.get("random") or {}).get(inten, {}).get("deposit_amounts") or (uni.get("random") or {}).get("balanced", {}).get("deposit_amounts") or []), dep_mn, dep_mx, include=[dep_amt0]) if "deposits" in vg else [dep_amt0]

    rows.append({"Dimension": "Funding cadence", "Baseline": dep_freq0, "Drift": fmt_enum(dep_freq_choices) if "deposits" in vg else "Pinned"})
    rows.append({"Dimension": "Deposit amount ($)", "Baseline": f"{dep_amt0:g}", "Drift": fmt_range(dep_amt_choices) if "deposits" in vg else "Pinned"})

    # -------------------------
    # Buys
    # -------------------------
    buy_freq0 = str(baseline_or("weekly", "buy_freq")).lower()
    buy_amt0 = float(baseline_or(50.0, "buy_amount_usd") or 0.0)

    if mode == "neighborhood":
        buy_freq_choices = _neighbor_enum(buy_freq0, list(buy_freqs_u), width=w) if "buys" in vg else [buy_freq0]
        buy_amt_choices = _neighbor_bins(buy_amt0, _filter_numeric(list(uni["money_bins"]), buy_mn, buy_mx, include=[buy_amt0]), width=w) if "buys" in vg else [buy_amt0]
    else:
        buy_freq_choices = list(buy_freqs_u) if "buys" in vg else [buy_freq0]
        buy_amt_choices = _filter_numeric(list((uni.get("random") or {}).get(inten, {}).get("buy_amounts") or (uni.get("random") or {}).get("balanced", {}).get("buy_amounts") or []), buy_mn, buy_mx, include=[buy_amt0]) if "buys" in vg else [buy_amt0]

    rows.append({"Dimension": "Buy cadence", "Baseline": buy_freq0, "Drift": fmt_enum(buy_freq_choices) if "buys" in vg else "Pinned"})
    rows.append({"Dimension": "Buy amount ($)", "Baseline": f"{buy_amt0:g}", "Drift": fmt_range(buy_amt_choices) if "buys" in vg else "Pinned"})

    # -------------------------
    # Entry gate
    # -------------------------
    if base_is_logic():
        base_gate = f"Logic builder ({len((bp.get('entry_logic') or {}).get('clauses') or []):d} clauses)"
        if "logic" in vg:
            drift_gate = f"Mutations near baseline (width={w})"
        else:
            drift_gate = "Pinned"
    else:
        base_gate = str(baseline_or("none", "buy_filter")).lower()
        if mode == "neighborhood":
            gate_choices = _neighbor_enum(base_gate, list(buy_filters_u), width=w) if "filter" in vg else [base_gate]
        else:
            gate_choices = list(buy_filters_u) if "filter" in vg else [base_gate]
        drift_gate = fmt_enum(gate_choices) if "filter" in vg else "Pinned"

    rows.append({"Dimension": "Entry gate type", "Baseline": base_gate, "Drift": drift_gate})

    # -------------------------
    # Allocation
    # -------------------------
    alloc0 = float(baseline_or(1.0, "max_alloc_pct") or 1.0)
    if mode == "neighborhood":
        alloc_choices = _neighbor_bins(alloc0, _filter_numeric(list(uni["alloc_bins"]), alloc_mn, alloc_mx, include=[alloc0]), width=w) if "alloc" in vg else [alloc0]
    else:
        alloc_choices = _filter_numeric(list((uni.get("random") or {}).get(inten, {}).get("max_alloc_pcts") or (uni.get("random") or {}).get("balanced", {}).get("max_alloc_pcts") or []), alloc_mn, alloc_mx, include=[alloc0]) if "alloc" in vg else [alloc0]
    rows.append({"Dimension": "Allocation cap (% equity)", "Baseline": f"{alloc0*100:.0f}%", "Drift": (fmt_range([x*100 for x in alloc_choices]) + "%") if "alloc" in vg else "Pinned"})

    # -------------------------
    # Risk & exits
    # -------------------------
    sl0 = float(baseline_or(0.0, "sl_pct") or 0.0)
    tp0 = float(baseline_or(0.0, "tp_pct") or 0.0)
    sell0 = float(baseline_or(1.0, "tp_sell_fraction") or 1.0)
    res0 = float(baseline_or(0.0, "reserve_frac_of_proceeds") or 0.0)
    trail0 = float(baseline_or(0.0, "trail_pct") or 0.0)
    hold0 = int(baseline_or(0, "max_hold_bars") or 0)

    if mode == "neighborhood":
        sl_choices = _neighbor_bins(sl0, _filter_numeric(list(uni["sl_bins"]), sl_mn, sl_mx, include=[sl0]), width=w) if "risk" in vg else [sl0]
        tp_choices = _neighbor_bins(tp0, _filter_numeric(list(uni["tp_bins"]), tp_mn, tp_mx, include=[tp0]), width=w) if "risk" in vg else [tp0]
        sell_choices = _neighbor_bins(sell0, _filter_numeric(list(uni["frac_bins"]), sell_mn, sell_mx, include=[sell0]), width=w) if "risk" in vg else [sell0]
        res_choices = _neighbor_bins(res0, _filter_numeric(list(uni["frac_bins"]), res_mn, res_mx, include=[res0]), width=w) if "risk" in vg else [res0]
        trail_choices = _neighbor_bins(trail0, _filter_numeric(list(uni["trail_bins"]), trail_mn, trail_mx, include=[trail0]), width=w) if "risk" in vg else [trail0]
        hold_choices = ([hold0] if ("risk" not in vg) else _filter_int(list(uni["hold_bins"]), hold_mn_i, hold_mx_i, include=[hold0]))
    else:
        r = (uni.get("random") or {}).get(inten, {}) or (uni.get("random") or {}).get("balanced", {}) or {}
        sl_choices = _filter_numeric(list(r.get("sl_pcts") or []), sl_mn, sl_mx, include=[sl0]) if "risk" in vg else [sl0]
        tp_choices = _filter_numeric(list(r.get("tp_pcts") or []), tp_mn, tp_mx, include=[tp0]) if "risk" in vg else [tp0]
        sell_choices = _filter_numeric(list(r.get("tp_sell_fracs") or []), sell_mn, sell_mx, include=[sell0]) if "risk" in vg else [sell0]
        res_choices = _filter_numeric(list(r.get("reserve_fracs") or []), res_mn, res_mx, include=[res0]) if "risk" in vg else [res0]
        trail_choices = _filter_numeric(list(r.get("trail_pcts") or []), trail_mn, trail_mx, include=[trail0]) if "risk" in vg else [trail0]
        hold_choices = _filter_int(list(r.get("max_hold_bars") or []), hold_mn_i, hold_mx_i, include=[hold0]) if "risk" in vg else [hold0]

    rows.append({"Dimension": "Stop loss (%)", "Baseline": f"{sl0*100:.0f}%", "Drift": (fmt_range([x*100 for x in sl_choices]) + "%") if "risk" in vg else "Pinned"})
    rows.append({"Dimension": "Take profit (%)", "Baseline": f"{tp0*100:.0f}%", "Drift": (fmt_range([x*100 for x in tp_choices]) + "%") if "risk" in vg else "Pinned"})
    rows.append({"Dimension": "TP sell fraction", "Baseline": f"{sell0:.2f}", "Drift": fmt_range(sell_choices) if "risk" in vg else "Pinned"})
    rows.append({"Dimension": "Reserve fraction", "Baseline": f"{res0:.2f}", "Drift": fmt_range(res_choices) if "risk" in vg else "Pinned"})
    rows.append({"Dimension": "Trailing stop (%)", "Baseline": f"{trail0*100:.0f}%", "Drift": (fmt_range([x*100 for x in trail_choices]) + "%") if "risk" in vg else "Pinned"})
    rows.append({"Dimension": "Max hold (bars)", "Baseline": str(hold0), "Drift": fmt_range([float(x) for x in hold_choices]) if "risk" in vg else "Pinned"})

    # Logic share (random mode only, when logic is allowed)
    if str(mode).lower() == "random" and ("logic" in vg) and ("filter" in vg):
        rows.append({"Dimension": "Logic share (random)", "Baseline": f"{logic_frac:.2f}", "Drift": "Controls mix of entry modes"})

    return rows


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate DCA/Swing grid (JSONL)")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--n", type=int, default=1000, help="Number of configs to generate")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--mode",
        default="random",
        choices=["random", "neighborhood"],
        help="random = broad grid; neighborhood = variations around --base",
    )
    ap.add_argument(
        "--base",
        default=None,
        help="Path to baseline config JSON (required for --mode neighborhood)",
    )
    ap.add_argument(
        "--include-base",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, include the baseline config as the first row (also works for random).",
    )
    ap.add_argument(
        "--width",
        default="medium",
        choices=["narrow", "medium", "wide"],
        help="Neighborhood breadth (only for --mode neighborhood).",
    )

    ap.add_argument(
        "--intensity",
        default="balanced",
        choices=["conservative", "balanced", "aggressive"],
        help="Drift intensity (how far parameters can move). For neighborhood mode this maps to width; for random mode it changes the sampling universe.",
    )
    ap.add_argument(
        "--vary",
        default="deposits,buys,filter,alloc,risk",
        help="Comma-separated groups allowed to vary (pins non-selected groups to baseline if --base is provided).",
    )
    ap.add_argument(
        "--overrides",
        default=None,
        help="Optional path to a JSON file that overrides drift ranges/options (min/max and option subsets).",
    )

    ap.add_argument(
        "--logic-frac",
        default=0.35,
        type=float,
        help="For --mode random: fraction (0-1) of configs that use the logic builder entry mode.",
    )

    args = ap.parse_args()

    out_path = Path(args.out)
    n = int(args.n)
    seed = int(args.seed)
    mode = str(args.mode)
    overrides = _load_overrides(getattr(args, "overrides", None))

    rows: List[Dict[str, Any]] = []

    if mode == "neighborhood":
        if not args.base:
            raise SystemExit("--base is required for --mode neighborhood")
        base_cfg = _read_json(Path(args.base))
    else:
        base_cfg = _read_json(Path(args.base)) if args.base else None

    intensity = _resolve_intensity(getattr(args, "intensity", "balanced"))
    vary_groups = {x.strip().lower() for x in str(args.vary).split(",") if x.strip()}

    if mode == "neighborhood":
        rows = generate_grid_neighborhood(
            n=n,
            seed=seed,
            base_cfg=base_cfg,
            width=str(args.width),
            intensity=intensity,
            vary_groups=vary_groups,
            overrides=overrides,
        )
    else:
        rows = generate_grid_random(
            n=n,
            seed=seed,
            logic_frac=float(args.logic_frac),
            intensity=intensity,
            base_cfg=base_cfg if isinstance(base_cfg, dict) else None,
            vary_groups=vary_groups,
            overrides=overrides,
        )

    if bool(args.include_base) and mode != "neighborhood":
        # Try to load a baseline from --base if provided, else skip.
        if args.base:
            bp = _normalize_base_params(_read_json(Path(args.base)))
            base_row = build_dca_cfg(
                deposit_freq=str(bp["deposit_freq"]),
                deposit_amount_usd=float(bp["deposit_amount_usd"]),
                buy_freq=str(bp["buy_freq"]),
                buy_amount_usd=float(bp["buy_amount_usd"]),
                buy_filter=str(bp["buy_filter"]),
                ema_len=int(bp.get("ema_len", 200)),
                rsi_thr=float(bp.get("rsi_thr", 40.0)),
                max_alloc_pct=float(bp["max_alloc_pct"]),
                sl_pct=float(bp["sl_pct"]),
                trail_pct=float(bp["trail_pct"]),
                max_hold_bars=int(bp["max_hold_bars"]),
                tp_pct=float(bp["tp_pct"]),
                tp_sell_fraction=float(bp["tp_sell_fraction"]),
                reserve_frac_of_proceeds=float(bp["reserve_frac_of_proceeds"]),
                entry_logic=bp.get("entry_logic"),
                is_baseline=True,
            )
            rows = [base_row] + rows

    _write_jsonl(out_path, rows)
    print(f"Wrote {len(rows):,} configs -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())