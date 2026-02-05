#!/usr/bin/env python
"""tools/generate_replay_artifacts.py

Generate replay artifacts (equity_curve.csv / trades.csv / fills.csv) for a given config_id
from an existing run directory.

Determinism goals:
- Prefer the EXACT df_feat used by the batch run (run_dir/df_feat.parquet) when present.
  This avoids tiny-but-real divergences from recomputing vol_bps/features.
- Default seed / starting_equity to the values recorded in batch_meta.json, unless explicitly provided.

This script is called by the Streamlit UI.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, is_dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np



def _add_contrib_profit_columns(eq_df: pd.DataFrame, *, starting_equity: float) -> pd.DataFrame:
    """Add contribution/profit clarity columns to an equity curve.

    Columns added:
      - contrib_total: cumulative cash-in (starting_equity + cumulative positive cashflow)
      - profit: equity - contrib_total

    Notes:
    - We intentionally derive this from the *recorded* per-bar cashflow rather than
      re-deriving deposit schedules from config params. That keeps results deterministic
      and auditable: equity_curve.csv is the source of truth for what happened.
    - If cashflow includes withdrawals (negative), contrib_total only counts cash-in.
    """
    if eq_df is None or eq_df.empty:
        return eq_df

    df = eq_df.copy()

    # cashflow is recorded per bar by the backtester (0 if none).
    if "cashflow" in df.columns:
        cf = pd.to_numeric(df["cashflow"], errors="coerce").fillna(0.0)
    else:
        cf = pd.Series([0.0] * len(df), index=df.index)

    deposits = cf.clip(lower=0.0)
    contrib_total = float(starting_equity) + deposits.cumsum()

    df["contrib_total"] = contrib_total

    if "equity" in df.columns:
        eq = pd.to_numeric(df["equity"], errors="coerce")
        df["profit"] = eq - df["contrib_total"]

    return df


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_repo_on_syspath() -> None:
    rr = str(_repo_root())
    if rr not in sys.path:
        sys.path.insert(0, rr)


def _import_symbol(dotted: str):
    """Import a symbol from 'module:attr' or 'module.attr'."""
    import importlib

    s = str(dotted or "").strip()
    if not s:
        raise ImportError("Empty symbol path")
    if ":" in s:
        mod_name, attr = s.split(":", 1)
    else:
        parts = s.split(".")
        if len(parts) < 2:
            raise ImportError(f"Invalid symbol path: {s}")
        mod_name, attr = ".".join(parts[:-1]), parts[-1]
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, attr, None)
    if obj is None:
        raise ImportError(f"Symbol not found: {s}")
    return obj


def _make_constraints() -> Any:
    """Mirror defaults from engine.batch for consistency."""
    from engine.contracts import EngineConstraints  # type: ignore

    return EngineConstraints(
        max_leverage=20.0,
        maint_margin_rate=0.005,
        price_tick=0.1,
        qty_step=0.001,
        min_notional_usdt=5.0,
    )


def _instantiate_template(template_cls: Any, cfg: Any) -> Any:
    try:
        return template_cls(config=cfg)
    except TypeError:
        return template_cls(cfg)


def _progress_write(path: Optional[Path], obj: Dict[str, Any]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
        df = df.sort_values("dt")
    return df


def _best_data_path(meta: Dict[str, Any]) -> Path:
    p = str(meta.get("data") or "").strip()
    if not p:
        raise FileNotFoundError("batch_meta.json missing 'data' path; cannot load dataset for replay.")
    return Path(p).expanduser().resolve()


def _ensure_vol_bps_like_batch(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Match engine.batch._ensure_vol_bps() to avoid divergences."""
    if "vol_bps" in df.columns:
        df = df.copy()
        df["vol_bps"] = pd.to_numeric(df["vol_bps"], errors="coerce").fillna(0.0)
        return df

    import numpy as np



def _add_contrib_profit_columns(eq_df: pd.DataFrame, *, starting_equity: float) -> pd.DataFrame:
    """Add contribution/profit clarity columns to an equity curve.

    Columns added:
      - contrib_total: cumulative cash-in (starting_equity + cumulative positive cashflow)
      - profit: equity - contrib_total

    Notes:
    - We intentionally derive this from the *recorded* per-bar cashflow rather than
      re-deriving deposit schedules from config params. That keeps results deterministic
      and auditable: equity_curve.csv is the source of truth for what happened.
    - If cashflow includes withdrawals (negative), contrib_total only counts cash-in.
    """
    if eq_df is None or eq_df.empty:
        return eq_df

    df = eq_df.copy()

    # cashflow is recorded per bar by the backtester (0 if none).
    if "cashflow" in df.columns:
        cf = pd.to_numeric(df["cashflow"], errors="coerce").fillna(0.0)
    else:
        cf = pd.Series([0.0] * len(df), index=df.index)

    deposits = cf.clip(lower=0.0)
    contrib_total = float(starting_equity) + deposits.cumsum()

    df["contrib_total"] = contrib_total

    if "equity" in df.columns:
        eq = pd.to_numeric(df["equity"], errors="coerce")
        df["profit"] = eq - df["contrib_total"]

    return df

    w = int(max(2, window))
    c = df["close"].astype(float)
    r = np.log(c).diff()
    vol = r.rolling(w, min_periods=max(2, w // 2)).std()
    out = df.copy()
    out["vol_bps"] = (vol.fillna(0.0) * 10_000.0).astype(float)
    return out


def _load_df_for_replay(run_dir: Path, meta: Dict[str, Any], progress_path: Optional[Path]) -> Tuple[pd.DataFrame, bool]:
    """Load the exact df_feat used by the run when possible."""
    df_feat_path = run_dir / "df_feat.parquet"
    if df_feat_path.exists():
        _progress_write(progress_path, {"stage": "replay", "phase": "load_df", "source": "df_feat.parquet"})
        return pd.read_parquet(df_feat_path), True

    # Fall back: load raw data and reproduce batch preprocessing as best as possible.
    data_path = _best_data_path(meta)
    _progress_write(progress_path, {"stage": "replay", "phase": "load_df", "source": str(data_path)})

    df = _load_df(data_path)

    # Ensure essential columns are numeric and sorted.
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()

    # Match vol_bps logic used by batch runner.
    vol_window = int(meta.get("vol_window", 60) or 60)
    df = _ensure_vol_bps_like_batch(df, window=vol_window)

    # If the original run computed features in-batch, do the same.
    features_ready = bool(meta.get("features_ready", False))
    if not features_ready:
        from engine.features import add_features  # type: ignore
        df = add_features(df)

    return df, True  # after this point, features are ready for the engine


def _load_cfg_obj_raw(run_dir: Path, cfg_id: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Load the raw config object for cfg_id, preferring configs_resolved.jsonl."""
    # Prefer configs_resolved.jsonl because it contains normalized/expanded configs.
    resolved = run_dir / "configs_resolved.jsonl"
    if resolved.exists():
        with resolved.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                if str(obj.get("config_id")) != str(cfg_id):
                    continue
                # Common shape: {config_id, line_no, normalized:{...}}
                norm = obj.get("normalized")
                if isinstance(norm, dict) and norm:
                    return norm
                return obj

    # Fall back: try to find config in grid by index if available.
    grid_path = str(meta.get("grid") or "").strip()
    if not grid_path:
        raise FileNotFoundError("Could not find configs_resolved.jsonl and batch_meta.json missing 'grid'.")

    gp = Path(grid_path).expanduser().resolve()
    if not gp.exists():
        raise FileNotFoundError(f"Grid not found: {gp}")

    with gp.open("r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict) and str(obj.get("config_id")) == str(cfg_id):
                # Some grids may already include config_id
                return obj

    raise FileNotFoundError(f"Could not locate config_id={cfg_id} in configs_resolved.jsonl or grid.")


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return dict(obj.__dict__)
        except Exception:
            pass
    return obj


def _build_engine_cfg(meta: Dict[str, Any]) -> Any:
    from engine.backtester import BacktestConfig  # type: ignore

    market_mode = str(meta.get("market_mode", "spot") or "spot").lower()
    raw = meta.get("engine_cfg")
    if not isinstance(raw, dict):
        return BacktestConfig(market_mode=market_mode)

    valid = {f.name for f in fields(BacktestConfig)}
    cfg_kwargs = {k: raw[k] for k in raw.keys() if k in valid}
    cfg_kwargs.setdefault("market_mode", market_mode)
    return BacktestConfig(**cfg_kwargs)


def _build_constraints(meta: Dict[str, Any]) -> Any:
    from engine.contracts import EngineConstraints  # type: ignore

    raw = meta.get("engine_constraints")
    if isinstance(raw, dict):
        keys = {"price_tick", "qty_step", "min_notional_usdt", "max_leverage", "maint_margin_rate"}
        if keys.issubset(set(raw.keys())):
            try:
                return EngineConstraints(
                    price_tick=float(raw["price_tick"]),
                    qty_step=float(raw["qty_step"]),
                    min_notional_usdt=float(raw["min_notional_usdt"]),
                    max_leverage=float(raw["max_leverage"]),
                    maint_margin_rate=float(raw["maint_margin_rate"]),
                )
            except Exception:
                pass
    return _make_constraints()


def _run_replay_once(
    *,
    df_feat: pd.DataFrame,
    cfg_obj_raw: Dict[str, Any],
    meta: Dict[str, Any],
    seed: int,
    starting_equity: float,
    progress_path: Optional[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Run a single config using engine.backtester.run_backtest_once."""
    from engine.backtester import run_backtest_once  # type: ignore
    from engine.contracts import StrategyConfig  # type: ignore

    template_path = str(meta.get("template") or "").strip()
    if not template_path:
        name = str(cfg_obj_raw.get("strategy_name") or "").strip().lower()
        template_path = (
            "strategies.dca_swing:Strategy"
            if name in {"dca", "dca_swing"}
            else "strategies.universal:UniversalStrategy"
        )

    _progress_write(progress_path, {"stage": "replay", "phase": "template", "template": template_path})
    template_cls = _import_symbol(template_path)

    # Parse/validate config using public parser if present.
    cfg_obj: Any = None
    try:
        from engine.batch import parse_strategy_config  # type: ignore

        _cfg_id2, cfg_obj, _norm = parse_strategy_config(cfg_obj_raw, line_no=0)  # type: ignore
    except Exception:
        try:
            cfg_obj = StrategyConfig(
                strategy_name=str(cfg_obj_raw.get("strategy_name") or "universal"),
                side=str(cfg_obj_raw.get("side") or "long"),
                params=dict(cfg_obj_raw.get("params") or {}),
            )
        except Exception:
            cfg_obj = cfg_obj_raw

    strategy = _instantiate_template(template_cls, cfg_obj)
    constraints = _build_constraints(meta)
    engine_cfg = _build_engine_cfg(meta)

    _progress_write(
        progress_path,
        {
            "stage": "replay",
            "phase": "run",
            "seed": int(seed),
            "starting_equity": float(starting_equity),
            "market_mode": str(getattr(engine_cfg, "market_mode", "spot")),
        },
    )

    metrics, fills_df, eq_df, trades_df, _guard = run_backtest_once(
        df=df_feat,
        strategy=strategy,
        seed=int(seed),
        starting_equity=float(starting_equity),
        constraints=constraints,
        cfg=engine_cfg,
        verbose=False,
        show_progress=False,
        features_ready=True,
        record_fills=True,
        record_equity_curve=True,
    )

    _progress_write(progress_path, {"stage": "replay", "phase": "done"})

    return eq_df, fills_df, trades_df, metrics


# =========================
# Events (price + actions)
# =========================

_EVENT_COLS = ["dt", "event", "side", "price", "qty", "reason", "detail"]


def _pick_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_dt_utc(x: Any) -> Optional[pd.Timestamp]:
    try:
        ts = pd.to_datetime(x, errors="coerce", utc=True)
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None


def _build_events_df(fills_df: Optional[pd.DataFrame], trades_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Best-effort event log derived from fills/trades.

    Produces events.csv used by the UI to overlay entries/exits/TPs on the price chart.
    This is NOT trading advice; it's just labeling what the backtest engine did.
    """
    events: List[Dict[str, Any]] = []

    if fills_df is None or fills_df.empty:
        return pd.DataFrame(columns=_EVENT_COLS)

    f = fills_df.copy()

    dt_col = _pick_first_col(f, ["dt", "fill_dt", "ts", "timestamp", "time"])
    side_col = _pick_first_col(f, ["side", "action"])
    price_col = _pick_first_col(f, ["price", "fill_price", "px"])
    qty_col = _pick_first_col(f, ["qty", "base_qty", "asset_qty", "filled_qty", "q"])

    reason_col = _pick_first_col(f, ["reason", "tag", "label", "message"])

    if dt_col is None or side_col is None:
        return pd.DataFrame(columns=_EVENT_COLS)

    # Normalize dt
    if dt_col == "ts":
        s = pd.to_numeric(f[dt_col], errors="coerce")
        mx = float(s.dropna().max()) if not s.dropna().empty else 0.0
        unit = "ms" if mx > 1e12 else "s"
        f["_dt"] = pd.to_datetime(s, unit=unit, errors="coerce", utc=True)
    else:
        f["_dt"] = pd.to_datetime(f[dt_col], errors="coerce", utc=True)

    f = f.dropna(subset=["_dt"]).sort_values("_dt")

    # Running position to classify partial sells as TP vs full exit.
    pos = 0.0
    eps = 1e-12

    for _, r in f.iterrows():
        dt = r["_dt"]
        side = str(r.get(side_col) or "").strip().lower()

        try:
            qty = float(r.get(qty_col)) if qty_col is not None else float("nan")
        except Exception:
            qty = float("nan")
        if not (qty == qty) or qty <= 0:
            qty = float("nan")

        try:
            price = float(r.get(price_col)) if price_col is not None else float("nan")
        except Exception:
            price = float("nan")
        if not (price == price):
            price = float("nan")

        reason = str(r.get(reason_col) or "").strip() if reason_col is not None else ""
        detail = ""

        is_buy = side in {"buy", "b", "long", "entry", "open"}
        is_sell = side in {"sell", "s", "exit", "close"}

        if not is_buy and not is_sell:
            continue

        before = pos
        if is_buy:
            pos = pos + (qty if qty == qty else 0.0)
            ev = "ENTRY" if before <= eps else "ADD"
        else:
            pos = pos - (qty if qty == qty else 0.0)
            ev = "TP" if pos > eps else "EXIT"

        events.append(
            {
                "dt": dt.isoformat(),
                "event": ev,
                "side": "buy" if is_buy else "sell",
                "price": None if not (price == price) else float(price),
                "qty": None if not (qty == qty) else float(qty),
                "reason": reason,
                "detail": detail,
            }
        )

    df_events = pd.DataFrame(events)

    if df_events.empty:
        return pd.DataFrame(columns=_EVENT_COLS)

    # Enrich sell events with exit reason from trades_df if available.
    if trades_df is not None and not trades_df.empty:
        t = trades_df.copy()
        exit_dt_col = _pick_first_col(t, ["exit_dt", "dt_exit", "exit_time", "exit_ts"])
        exit_reason_col = _pick_first_col(t, ["exit_reason", "reason", "exit_reason_detail", "exit_type"])
        if exit_dt_col is not None and exit_reason_col is not None:
            t["_exit_dt"] = pd.to_datetime(t[exit_dt_col], errors="coerce", utc=True)
            t = t.dropna(subset=["_exit_dt"]).sort_values("_exit_dt")
            if not t.empty:
                exit_ns = t["_exit_dt"].astype("int64").to_numpy()
                exit_reason = t[exit_reason_col].astype(str).to_numpy()

                # Update EXIT/TP reasons by nearest exit_dt within tolerance.
                tol_ns = int(2 * 24 * 3600 * 1e9)  # 2 days
                ev_dt = pd.to_datetime(df_events["dt"], errors="coerce", utc=True)
                ev_ns = ev_dt.astype("int64").to_numpy()

                for i_ev in range(len(df_events)):
                    if df_events.loc[i_ev, "event"] not in {"TP", "EXIT"}:
                        continue
                    ns = int(ev_ns[i_ev])
                    j = int(np.argmin(np.abs(exit_ns - ns))) if exit_ns.size else -1
                    if j >= 0 and abs(int(exit_ns[j]) - ns) <= tol_ns:
                        rr = str(exit_reason[j] or "").strip()
                        if rr:
                            df_events.loc[i_ev, "reason"] = rr

    # Ensure columns + stable order
    for c in _EVENT_COLS:
        if c not in df_events.columns:
            df_events[c] = None
    df_events = df_events[_EVENT_COLS].copy()
    return df_events

def main() -> int:
    _ensure_repo_on_syspath()

    ap = argparse.ArgumentParser()
    ap.add_argument("--from-run", required=True, type=str)
    ap.add_argument("--config-id", required=True, type=str)
    ap.add_argument("--out-dir", default="", type=str)
    ap.add_argument("--seed", default=None, type=int, help="Defaults to batch_meta.json seed")
    ap.add_argument("--starting-equity", default=None, type=float, help="Defaults to batch_meta.json starting_equity")
    ap.add_argument("--progress-file", default="", type=str)
    args = ap.parse_args()

    run_dir = Path(args.from_run).resolve()
    cfg_id = args.config_id.strip()

    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"--from-run must be a run directory: {run_dir}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (run_dir / "replay_cache" / cfg_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_path = Path(args.progress_file).resolve() if str(args.progress_file).strip() else None

    meta_path = run_dir / "batch_meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = _read_json(meta_path)
        except Exception:
            meta = {}

    # Default deterministic knobs from meta unless user explicitly overrides.
    seed = int(args.seed) if args.seed is not None else int(meta.get("seed", 1) or 1)
    starting_equity = float(args.starting_equity) if args.starting_equity is not None else float(meta.get("starting_equity", 10_000.0) or 10_000.0)

    t0 = time.time()
    _progress_write(progress_path, {"stage": "replay", "phase": "init", "out_dir": str(out_dir)})

    df_feat, _feat_ready = _load_df_for_replay(run_dir, meta, progress_path)
    cfg_obj_raw = _load_cfg_obj_raw(run_dir, cfg_id, meta)

    eq_df, fills_df, trades_df, metrics = _run_replay_once(
        df_feat=df_feat,
        cfg_obj_raw=cfg_obj_raw,
        meta=meta,
        seed=seed,
        starting_equity=starting_equity,
        progress_path=progress_path,
    )

    # Write artifacts
    if fills_df is not None and not fills_df.empty:
        fills_df.to_csv(out_dir / "fills.csv", index=False)
    if trades_df is not None and not trades_df.empty:
        trades_df.to_csv(out_dir / "trades.csv", index=False)
    if eq_df is not None and not eq_df.empty:
        eq_df = _add_contrib_profit_columns(eq_df, starting_equity=float(starting_equity))
        eq_df.to_csv(out_dir / "equity_curve.csv", index=False)

    (out_dir / "metrics.json").write_text(json.dumps(_to_jsonable(metrics), indent=2), encoding="utf-8")

    # Write event tape for UI overlays (price + actions)
    try:
        events_df = _build_events_df(fills_df, trades_df)
        # Always write the file so the UI can detect regeneration, even if empty.
        events_df.to_csv(out_dir / "events.csv", index=False)
    except Exception as e:
        # Don't fail replay artifacts if event tape generation breaks.
        try:
            pd.DataFrame(columns=_EVENT_COLS).to_csv(out_dir / "events.csv", index=False)
        except Exception:
            pass
        _progress_write(progress_path, {"stage": "replay", "phase": "events_error", "err": str(e)})

    _progress_write(progress_path, {"stage": "replay", "phase": "write", "sec": float(time.time() - t0)})
    print(f"Replay artifacts written: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())