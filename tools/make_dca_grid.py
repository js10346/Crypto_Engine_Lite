from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


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


def generate_grid(
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate DCA/Swing grid (JSONL)")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--n", type=int, default=1000, help="Number of configs to generate")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rows = generate_grid(n=int(args.n), seed=int(args.seed))
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