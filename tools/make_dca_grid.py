from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


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


def _neighbor_bins(
    baseline: float,
    bins: Sequence[float],
    *,
    width: str,
) -> List[float]:
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
    # stable unique
    out2 = sorted({float(x) for x in out})
    return out2


def _neighbor_enum(
    baseline: str,
    options: Sequence[str],
    *,
    width: str,
) -> List[str]:
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
    if len(opts_l) <= 1:
        return [b]
    nxt = opts_l[(i + 1) % len(opts_l)]
    if nxt == b:
        return [b]
    return [b, nxt]


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

    # Defaults mirror strategies.dca_swing.DEFAULT_CONFIG
    p: Dict[str, Any] = {
        "deposit_freq": _as_str(params_in.get("deposit_freq"), "weekly").lower(),
        "deposit_amount_usd": _as_float(params_in.get("deposit_amount_usd"), 50.0),
        "buy_freq": _as_str(params_in.get("buy_freq"), "weekly").lower(),
        "buy_amount_usd": _as_float(params_in.get("buy_amount_usd"), 50.0),
        "buy_filter": _as_str(params_in.get("buy_filter"), "none").lower(),
        "ema_len": _as_int(params_in.get("ema_len"), 200),
        "rsi_thr": _as_float(params_in.get("rsi_thr"), 40.0),
        "max_alloc_pct": _as_float(params_in.get("max_alloc_pct"), 1.0),
        "sl_pct": _as_float(params_in.get("sl_pct"), 0.0),
        "tp_pct": _as_float(params_in.get("tp_pct"), 0.0),
        "tp_sell_fraction": _as_float(params_in.get("tp_sell_fraction"), 0.50),
        "reserve_frac_of_proceeds": _as_float(
            params_in.get("reserve_frac_of_proceeds"), 0.50
        ),
    }

    # Basic cleanup
    if p["deposit_freq"] in {"none", "off", "0", ""}:
        p["deposit_freq"] = "none"
        p["deposit_amount_usd"] = 0.0

    if p["buy_filter"] not in {"none", "below_ema", "rsi_below"}:
        p["buy_filter"] = "none"

    return p

def build_dca_cfg(
    *,
    deposit_freq: str,
    deposit_amount_usd: float,
    buy_freq: str,
    buy_amount_usd: float,
    buy_filter: str,
    ema_len: Optional[int],
    rsi_thr: Optional[float],
    max_alloc_pct: float,
    sl_pct: float,
    tp_pct: float,
    tp_sell_fraction: float,
    reserve_frac_of_proceeds: float,
    is_baseline: bool = False,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "deposit_freq": str(deposit_freq),
        "deposit_amount_usd": float(deposit_amount_usd),
        "buy_freq": str(buy_freq),
        "buy_amount_usd": float(buy_amount_usd),
        "buy_filter": str(buy_filter),
        "max_alloc_pct": float(_clamp01(max_alloc_pct)),
        "sl_pct": float(max(0.0, sl_pct)),
        "tp_pct": float(max(0.0, tp_pct)),
        "tp_sell_fraction": float(_clamp01(tp_sell_fraction)),
        "reserve_frac_of_proceeds": float(_clamp01(reserve_frac_of_proceeds)),
    }

    if bool(is_baseline):
        params["__baseline__"] = True

    if buy_filter == "below_ema":
        params["ema_len"] = int(ema_len or 200)
    if buy_filter == "rsi_below":
        params["rsi_thr"] = float(rsi_thr or 40.0)

    # If TP is disabled, make sell/reserve effectively no-op.
    if params["tp_pct"] <= 0.0:
        params["tp_sell_fraction"] = 1.0
        params["reserve_frac_of_proceeds"] = 0.0

    # If there is no selling, reserve is pointless.
    if params["tp_sell_fraction"] <= 0.0:
        params["reserve_frac_of_proceeds"] = 0.0

    return {
        "strategy_name": "dca_swing",
        "side": "long",
        "params": params,
    }


def generate_grid_random(
    *,
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(int(seed))

    deposit_freqs = ["none", "weekly", "monthly"]
    deposit_amounts = [25.0, 50.0, 100.0, 200.0]

    buy_freqs = ["weekly", "monthly"]
    buy_amounts = [25.0, 50.0, 100.0, 200.0]

    buy_filters = ["none", "below_ema", "rsi_below"]
    ema_lens = [50, 100, 200]
    rsi_thrs = [30.0, 35.0, 40.0, 45.0]

    max_alloc_pcts = [0.5, 0.75, 0.9, 1.0]

    sl_pcts = [0.0, 0.10, 0.15, 0.20, 0.25]
    tp_pcts = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    tp_sell_fracs = [0.25, 0.50, 1.0]
    reserve_fracs = [0.0, 0.25, 0.50, 0.75, 1.0]

    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    attempts = 0
    max_attempts = max(5000, int(n) * 50)

    while len(out) < int(n) and attempts < max_attempts:
        attempts += 1

        deposit_freq = str(_pick(rng, deposit_freqs))
        if deposit_freq == "none":
            deposit_amount = 0.0
        else:
            deposit_amount = float(_pick(rng, deposit_amounts))

        buy_freq = str(_pick(rng, buy_freqs))
        buy_amount = float(_pick(rng, buy_amounts))

        buy_filter = str(_pick(rng, buy_filters))
        ema_len = int(_pick(rng, ema_lens)) if buy_filter == "below_ema" else None
        rsi_thr = float(_pick(rng, rsi_thrs)) if buy_filter == "rsi_below" else None

        max_alloc_pct = float(_pick(rng, max_alloc_pcts))
        sl_pct = float(_pick(rng, sl_pcts))
        tp_pct = float(_pick(rng, tp_pcts))

        tp_sell_fraction = float(_pick(rng, tp_sell_fracs))
        reserve_frac = float(_pick(rng, reserve_fracs))

        cfg = build_dca_cfg(
            deposit_freq=deposit_freq,
            deposit_amount_usd=deposit_amount,
            buy_freq=buy_freq,
            buy_amount_usd=buy_amount,
            buy_filter=buy_filter,
            ema_len=ema_len,
            rsi_thr=rsi_thr,
            max_alloc_pct=max_alloc_pct,
            sl_pct=sl_pct,
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
    vary_groups: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Neighborhood generator around a baseline config.

    - Always includes the baseline config as the first row.
    - Samples discrete neighbor bins based on width.
    - vary_groups controls which groups are allowed to vary:
        {"deposits","buys","filter","alloc","risk"}
      If None: all groups vary.
    """
    rng = random.Random(int(seed))
    w = str(width).strip().lower()
    if w not in {"narrow", "medium", "wide"}:
        w = "medium"

    if vary_groups is None:
        vary_groups = {"deposits", "buys", "filter", "alloc", "risk"}
    else:
        vary_groups = {str(x).strip().lower() for x in vary_groups if str(x).strip()}

    bp = _normalize_base_params(base_cfg)

    # Canonical baseline config (enforces TP rules).
    base_row = build_dca_cfg(
        deposit_freq=str(bp["deposit_freq"]),
        deposit_amount_usd=float(bp["deposit_amount_usd"]),
        buy_freq=str(bp["buy_freq"]),
        buy_amount_usd=float(bp["buy_amount_usd"]),
        buy_filter=str(bp["buy_filter"]),
        ema_len=int(bp["ema_len"]) if bp["buy_filter"] == "below_ema" else None,
        rsi_thr=float(bp["rsi_thr"]) if bp["buy_filter"] == "rsi_below" else None,
        max_alloc_pct=float(bp["max_alloc_pct"]),
        sl_pct=float(bp["sl_pct"]),
        tp_pct=float(bp["tp_pct"]),
        tp_sell_fraction=float(bp["tp_sell_fraction"]),
        reserve_frac_of_proceeds=float(bp["reserve_frac_of_proceeds"]),
        is_baseline=True,
    )

    base_params = dict(base_row.get("params") or {})

    # Discrete universes (small, stable)
    deposit_freqs = ["none", "weekly", "monthly"]
    buy_freqs = ["weekly", "monthly"]
    buy_filters = ["none", "below_ema", "rsi_below"]

    # Wider money bins than the old random generator, but still reasonable for spot.
    money_bins = [0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0]
    alloc_bins = [0.25, 0.50, 0.75, 0.90, 1.0]
    sl_bins = [0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    tp_bins = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00, 1.50, 2.00]
    frac_bins = [0.0, 0.25, 0.50, 0.75, 1.0]
    ema_lens = [50, 100, 200]
    rsi_thrs = [25.0, 30.0, 35.0, 40.0, 45.0, 50.0]

    def fixed_or(var: str, values: List[Any], group: str) -> List[Any]:
        if group in vary_groups:
            return values
        return [var]

    dep_freq_choices = fixed_or(
        str(base_params.get("deposit_freq", "weekly")).lower(),
        _neighbor_enum(str(base_params.get("deposit_freq", "weekly")), deposit_freqs, width=w),
        "deposits",
    )
    dep_amt_choices = fixed_or(
        float(base_params.get("deposit_amount_usd", 0.0) or 0.0),
        _neighbor_bins(float(base_params.get("deposit_amount_usd", 0.0) or 0.0), money_bins, width=w),
        "deposits",
    )

    buy_freq_choices = fixed_or(
        str(base_params.get("buy_freq", "weekly")).lower(),
        _neighbor_enum(str(base_params.get("buy_freq", "weekly")), buy_freqs, width=w),
        "buys",
    )
    buy_amt_choices = fixed_or(
        float(base_params.get("buy_amount_usd", 0.0) or 0.0),
        _neighbor_bins(float(base_params.get("buy_amount_usd", 0.0) or 0.0), money_bins, width=w),
        "buys",
    )

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

    alloc_choices = fixed_or(
        float(base_params.get("max_alloc_pct", 1.0) or 1.0),
        _neighbor_bins(float(base_params.get("max_alloc_pct", 1.0) or 1.0), alloc_bins, width=w),
        "alloc",
    )

    sl_choices = fixed_or(
        float(base_params.get("sl_pct", 0.0) or 0.0),
        _neighbor_bins(float(base_params.get("sl_pct", 0.0) or 0.0), sl_bins, width=w),
        "risk",
    )
    tp_choices = fixed_or(
        float(base_params.get("tp_pct", 0.0) or 0.0),
        _neighbor_bins(float(base_params.get("tp_pct", 0.0) or 0.0), tp_bins, width=w),
        "risk",
    )
    sell_choices = fixed_or(
        float(base_params.get("tp_sell_fraction", 1.0) or 1.0),
        _neighbor_bins(float(base_params.get("tp_sell_fraction", 1.0) or 1.0), frac_bins, width=w),
        "risk",
    )
    reserve_choices = fixed_or(
        float(base_params.get("reserve_frac_of_proceeds", 0.0) or 0.0),
        _neighbor_bins(float(base_params.get("reserve_frac_of_proceeds", 0.0) or 0.0), frac_bins, width=w),
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
        ema_len = int(_pick(rng, list(ema_choices))) if buy_filter == "below_ema" else None
        rsi_thr = float(_pick(rng, list(rsi_choices))) if buy_filter == "rsi_below" else None

        max_alloc_pct = float(_pick(rng, list(alloc_choices)))
        sl_pct = float(_pick(rng, list(sl_choices)))
        tp_pct = float(_pick(rng, list(tp_choices)))
        tp_sell_fraction = float(_pick(rng, list(sell_choices)))
        reserve_frac = float(_pick(rng, list(reserve_choices)))
        cfg = build_dca_cfg(
            deposit_freq=deposit_freq,
            deposit_amount_usd=deposit_amount,
            buy_freq=buy_freq,
            buy_amount_usd=buy_amount,
            buy_filter=buy_filter,
            ema_len=ema_len,
            rsi_thr=rsi_thr,
            max_alloc_pct=max_alloc_pct,
            sl_pct=sl_pct,
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
        "--vary",
        default="deposits,buys,filter,alloc,risk",
        help=(
            "Comma-separated groups allowed to vary for neighborhood mode: "
            "deposits,buys,filter,alloc,risk. Default: all."
        ),
    )
    args = ap.parse_args()

    mode = str(args.mode).strip().lower()
    if mode == "neighborhood":
        if not args.base:
            raise SystemExit("--base is required when --mode neighborhood")
        base_path = Path(args.base).resolve()
        if not base_path.exists():
            raise SystemExit(f"--base not found: {base_path}")
        base_cfg = _read_json(base_path)
        vary_groups = {
            s.strip().lower()
            for s in str(args.vary or "").split(",")
            if s.strip()
        }
        rows = generate_grid_neighborhood(
            n=int(args.n),
            seed=int(args.seed),
            base_cfg=base_cfg,
            width=str(args.width),
            vary_groups=vary_groups,
        )
    else:
        # Random grid (optionally include baseline plan as first row)
        base_cfg: Optional[Dict[str, Any]] = None
        base_row: Optional[Dict[str, Any]] = None
        include_base = bool(getattr(args, "include_base", False))

        if args.base:
            base_path = Path(args.base).resolve()
            if not base_path.exists():
                raise SystemExit(f"--base not found: {base_path}")
            base_cfg = _read_json(base_path)

        if include_base:
            if base_cfg is None:
                raise SystemExit("--include-base requires --base")
            bp = _normalize_base_params(base_cfg)
            base_row = build_dca_cfg(
                deposit_freq=str(bp["deposit_freq"]),
                deposit_amount_usd=float(bp["deposit_amount_usd"]),
                buy_freq=str(bp["buy_freq"]),
                buy_amount_usd=float(bp["buy_amount_usd"]),
                buy_filter=str(bp["buy_filter"]),
                ema_len=int(bp["ema_len"]) if bp["buy_filter"] == "below_ema" else None,
                rsi_thr=float(bp["rsi_thr"]) if bp["buy_filter"] == "rsi_below" else None,
                max_alloc_pct=float(bp["max_alloc_pct"]),
                sl_pct=float(bp["sl_pct"]),
                tp_pct=float(bp["tp_pct"]),
                tp_sell_fraction=float(bp["tp_sell_fraction"]),
                reserve_frac_of_proceeds=float(bp["reserve_frac_of_proceeds"]),
                is_baseline=True,
            )

        n_total = int(args.n)
        n_rand = max(0, n_total - (1 if include_base else 0))
        rows_rand = generate_grid_random(n=int(n_rand), seed=int(args.seed))
        if include_base and base_row is not None:
            rows = [base_row, *rows_rand]
        else:
            rows = rows_rand

    out_path = Path(args.out).resolve()
    written = _write_jsonl(out_path, rows)

    print(f"Wrote {written} configs to: {out_path}")
    if written < int(args.n):
        print(
            f"Note: requested n={int(args.n)}, generated={written}. "
            "Increase max_attempts or reduce grid constraints."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())