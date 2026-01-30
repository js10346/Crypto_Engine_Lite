from __future__ import annotations

"""Generate replay artifacts for a single config_id.

Place this file at: `tools/generate_replay_artifacts.py`

Usage:
  python tools/generate_replay_artifacts.py --from-run runs/batch_... --config-id cfg_... [--out ...]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# --- Make repo root importable when running as a script (Windows-friendly) ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ---------------------------------------------------------------------------

from engine.backtester import (  # noqa: E402
    BacktestConfig,
    _build_cashflow_performance_stats,
    _build_performance_stats,
    run_backtest_once,
)
from engine.features import add_features  # noqa: E402
from engine.batch import (  # noqa: E402
    _ensure_vol_bps,
    _import_symbol,
    _instantiate_template,
    _load_ohlcv,
    _make_constraints,
    _normalize_columns,
    parse_strategy_config,
)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate replay artifacts for a single config_id")
    ap.add_argument("--from-run", required=True, help="Run directory produced by engine.batch")
    ap.add_argument("--config-id", required=True, help="config_id to render")
    ap.add_argument("--out", default=None, help="Output folder (default: <run>/replay_cache/<config_id>)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing artifacts if present")
    args = ap.parse_args(argv)

    run_dir = Path(args.from_run).resolve()
    cfg_id = str(args.config_id).strip()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    meta = _read_json(run_dir / "batch_meta.json")
    template = str(meta.get("template") or "strategies.universal:UniversalStrategy")
    market_mode = str(meta.get("market_mode") or "spot").lower()
    seed = int(meta.get("seed") or 1)
    starting_equity = float(meta.get("starting_equity") or 1000.0)
    vol_window = int(meta.get("vol_window") or 60)
    features_ready = bool(meta.get("features_ready") or False)

    out_dir = Path(args.out).resolve() if args.out else (run_dir / "replay_cache" / cfg_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    sentinel = out_dir / "equity_curve.csv"
    if sentinel.exists() and (not args.force):
        print(f"Artifacts already exist: {out_dir} (use --force to overwrite)")
        return 0

    # Locate normalized config
    resolved_path = run_dir / "configs_resolved.jsonl"
    if not resolved_path.exists():
        raise FileNotFoundError(f"Missing configs_resolved.jsonl: {resolved_path}")

    rows = _read_jsonl(resolved_path)
    norm: Dict[str, Any] = {}
    for r in rows:
        if str(r.get("config_id")) == cfg_id:
            norm = r.get("normalized") or {}
            break
    if not norm:
        raise ValueError(f"config_id not found in configs_resolved.jsonl: {cfg_id}")

    # Re-parse into StrategyConfig
    cid2, cfg, _norm2 = parse_strategy_config(norm, 1)
    if str(cid2) != cfg_id:
        # This should never happen, but be explicit if it does.
        raise ValueError(f"Resolved config hash mismatch: requested {cfg_id}, parsed {cid2}")

    # Load df_feat
    df_feat_path = run_dir / "df_feat.parquet"
    df_feat = None
    if df_feat_path.exists():
        t0 = time.time()
        df_feat = pd.read_parquet(df_feat_path)
        print(f"Loaded df_feat: {df_feat_path} ({time.time()-t0:.2f}s)")
    else:
        data_path = meta.get("data")
        if not data_path:
            raise FileNotFoundError("df_feat.parquet missing and batch_meta.json has no 'data' path")
        df = _load_ohlcv(str(data_path))
        df = _normalize_columns(df)
        df = _ensure_vol_bps(df, window=vol_window)
        if "liq_mult" not in df.columns:
            df["liq_mult"] = 1.0
        else:
            df["liq_mult"] = pd.to_numeric(df["liq_mult"], errors="coerce").fillna(1.0)
        df_feat = df if features_ready else add_features(df)

    assert df_feat is not None

    # Instantiate strategy
    template_cls = _import_symbol(template)
    strategy = _instantiate_template(template_cls, cfg)
    constraints = _make_constraints()
    engine_cfg = BacktestConfig(market_mode=market_mode)

    metrics, fills_df, eq_df, trades_df, _guard = run_backtest_once(
        df=df_feat,
        strategy=strategy,
        seed=seed,
        starting_equity=starting_equity,
        constraints=constraints,
        cfg=engine_cfg,
        show_progress=False,
        features_ready=True,
        record_fills=True,
        record_equity_curve=True,
    )

    # Add a performance block consistent with market mode
    try:
        if market_mode == "spot":
            perf = _build_cashflow_performance_stats(df_feat, eq_df)
        else:
            perf = _build_performance_stats(df_feat, eq_df)
        if isinstance(metrics, dict):
            metrics["performance"] = perf
    except Exception:
        pass

    # Write artifacts
    (out_dir / "config.json").write_text(json.dumps(norm, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if fills_df is not None and not fills_df.empty:
        fills_df.to_csv(out_dir / "fills.csv", index=False)
    if trades_df is not None and not trades_df.empty:
        trades_df.to_csv(out_dir / "trades.csv", index=False)
    if eq_df is not None and not eq_df.empty:
        eq_df.to_csv(out_dir / "equity_curve.csv", index=False)

    print(f"Wrote replay artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
