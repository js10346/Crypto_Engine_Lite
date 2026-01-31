#!/usr/bin/env python
"""
Generate replay artifacts (equity_curve.csv / trades.csv / fills.csv) for a given config_id
from an existing run directory.

Design goals:
- Avoid importing private helpers from engine.batch (they change a lot).
- Prefer configs_resolved.jsonl when present (already-expanded configs).
- Fall back to grid + config_index when needed, with best-effort grid-path inference.

This script is called by the Streamlit UI.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


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


def _make_constraints():
    # Mirror defaults from engine.batch for consistency.
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


def _tail_jsonl(path: Path, max_lines: int = 50_000) -> Iterable[Dict[str, Any]]:
    # jsonl files here are usually small (<10k lines), so keep it simple + robust.
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj
            if i >= max_lines:
                break


def _find_obj_in_jsonl(path: Path, cfg_id: str) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    for obj in _tail_jsonl(path):
        val = obj.get("config_id") or obj.get("cfg_id") or obj.get("id")
        if isinstance(val, str) and val == cfg_id:
            return obj
    return None


def _unwrap_cfg_obj(obj: Dict[str, Any], cfg_id: str) -> Dict[str, Any]:
    """configs_resolved.jsonl stores rows like {config_id, line_no, normalized:{...}}.
    We want the actual normalized strategy dict for replay."""
    if not isinstance(obj, dict):
        return {"config_id": str(cfg_id)}
    norm = obj.get("normalized")
    if isinstance(norm, dict):
        out = dict(norm)
        # Keep ids for provenance (harmless extra keys)
        out.setdefault("config_id", obj.get("config_id") or cfg_id)
        out.setdefault("line_no", obj.get("line_no"))
        return out
    # Some historical formats store under "cfg" or "config"
    for k in ("cfg", "config"):
        v = obj.get(k)
        if isinstance(v, dict):
            out = dict(v)
            out.setdefault("config_id", obj.get("config_id") or cfg_id)
            return out
    return obj

def _parse_run_timestamp(run_dir: Path) -> Optional[time.struct_time]:
    # run names look like batch_YYYYMMDD_HHMMSS_...
    parts = run_dir.name.split("_")
    if len(parts) >= 3 and len(parts[1]) == 8 and len(parts[2]) == 6:
        try:
            return time.strptime(parts[1] + parts[2], "%Y%m%d%H%M%S")
        except Exception:
            return None
    return None


def _infer_grid_path(run_dir: Path, meta: Dict[str, Any]) -> Optional[Path]:
    # 1) metadata hints
    for k in ("grid_path", "grid", "grid_file", "grid_jsonl"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            p = Path(v)
            if p.exists() and p.is_file():
                return p
            if p.exists() and p.is_dir():
                # find a likely grid file in that directory
                cands = sorted(p.glob("grid*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)
                if cands:
                    return cands[0]

    # 2) run dir itself
    cands = sorted(run_dir.glob("grid*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)
    if cands:
        return cands[0]

    # 3) .ui_tmp heuristic (best effort)
    rr = _repo_root()
    ui_tmp = rr / ".ui_tmp"
    if ui_tmp.exists():
        # Prefer run_* folders near the batch run timestamp
        run_ts = _parse_run_timestamp(run_dir)
        run_epoch = time.mktime(run_ts) if run_ts else None

        best: Tuple[float, Optional[Path]] = (1e100, None)
        for d in ui_tmp.glob("run_*"):
            if not d.is_dir():
                continue
            grids = list(d.glob("grid*.jsonl"))
            if not grids:
                continue
            # If we can compare times, pick nearest
            if run_epoch is not None:
                try:
                    dt = abs(d.stat().st_mtime - run_epoch)
                except Exception:
                    dt = 1e99
            else:
                dt = 0.0
            # Prefer folders closer in time; within that, newest grid
            if dt < best[0]:
                grids_sorted = sorted(grids, key=lambda x: x.stat().st_mtime, reverse=True)
                best = (dt, grids_sorted[0])
        if best[1] is not None:
            return best[1]

    return None


def _read_grid_obj_at_line(grid_path: Path, line_no: int) -> Dict[str, Any]:
    # line_no is 0-indexed
    with grid_path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i == line_no:
                return json.loads(line)
    raise FileNotFoundError(f"Line {line_no} not found in grid file: {grid_path}")


def _load_cfg_obj_raw(run_dir: Path, cfg_id: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    # Prefer resolved configs: stable + avoids template resolution.
    resolved = run_dir / "configs_resolved.jsonl"
    obj = _find_obj_in_jsonl(resolved, cfg_id)
    if obj is not None:
        return _unwrap_cfg_obj(obj, cfg_id)

    # Next: some runs may have configs.jsonl
    obj = _find_obj_in_jsonl(run_dir / "configs.jsonl", cfg_id)
    if obj is not None:
        return _unwrap_cfg_obj(obj, cfg_id)

    # Fallback: config_index + grid file (legacy)
    idx_path = run_dir / "config_index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing configs_resolved.jsonl and no config_index.json at: {idx_path}")

    idx = _read_json(idx_path)

    # index format: either {cfg_id: line_no} OR {"index": {...}, "grid_path": "..."} etc.
    line_no: Optional[int] = None
    if cfg_id in idx:
        v = idx[cfg_id]
        if isinstance(v, int):
            line_no = v
        elif isinstance(v, dict) and isinstance(v.get("line_no"), int):
            line_no = int(v["line_no"])
        elif isinstance(v, dict) and isinstance(v.get("line"), int):
            line_no = int(v["line"])
    elif isinstance(idx.get("index"), dict) and cfg_id in idx["index"]:
        v = idx["index"][cfg_id]
        if isinstance(v, int):
            line_no = v
        elif isinstance(v, dict) and isinstance(v.get("line"), int):
            line_no = int(v["line"])

    if line_no is None:
        raise KeyError(f"Config id not found in config_index.json: {cfg_id}")

    # grid path: from meta, index, or inferred
    grid_path: Optional[Path] = None
    if isinstance(idx.get("grid_path"), str):
        p = Path(idx["grid_path"])
        if p.exists():
            grid_path = p
    if grid_path is None:
        grid_path = _infer_grid_path(run_dir, meta)
    if grid_path is None or not grid_path.exists() or not grid_path.is_file():
        raise FileNotFoundError(
            "Could not locate grid JSONL file for replay.\n"
            f"- from-run: {run_dir}\n"
            f"- meta grid hints: {meta.get('grid_path') or meta.get('grid')}\n"
            f"- looked in: {run_dir}, and repo_root/.ui_tmp\n"
        )

    return _read_grid_obj_at_line(grid_path, int(line_no))


def _best_data_path(meta: Dict[str, Any]) -> Path:
    for k in ("data_path", "data", "data_file"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return Path(v)
    raise KeyError("batch_meta.json is missing a data path (expected data_path or data)")


def _load_df(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if data_path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
        df = df.sort_values("dt")
    return df


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


def _run_replay_once(
    *,
    df: pd.DataFrame,
    cfg_obj_raw: Dict[str, Any],
    meta: Dict[str, Any],
    seed: int,
    starting_equity: float,
    progress_path: Optional[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Run a single config using the public engine API (run_backtest_once)."""
    from engine.backtester import BacktestConfig, run_backtest_once  # type: ignore
    from engine.features import add_features  # type: ignore
    from engine.contracts import StrategyConfig  # type: ignore

    template_path = str(meta.get("template") or "").strip()
    if not template_path:
        name = str(cfg_obj_raw.get("strategy_name") or "").strip().lower()
        template_path = "strategies.dca_swing:Strategy" if name in {"dca", "dca_swing"} else "strategies.universal:UniversalStrategy"

    _progress_write(progress_path, {"stage": "replay", "phase": "template", "done": 1, "total": 5, "template": template_path})

    template_cls = _import_symbol(template_path)

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

    _progress_write(progress_path, {"stage": "replay", "phase": "features", "done": 2, "total": 5})

    features_ready = bool(meta.get("features_ready", False))
    df_feat = df if features_ready else add_features(df)

    if "vol_bps" not in df_feat.columns:
        df_feat = df_feat.copy()
        df_feat["vol_bps"] = (df_feat["close"].pct_change().abs().fillna(0.0) * 10000.0)

    _progress_write(progress_path, {"stage": "replay", "phase": "run", "done": 3, "total": 5})

    strategy = _instantiate_template(template_cls, cfg_obj)
    constraints = _make_constraints()

    market_mode = str(meta.get("market_mode", "spot") or "spot").lower()
    engine_cfg = BacktestConfig(market_mode=market_mode)

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

    _progress_write(progress_path, {"stage": "replay", "phase": "run", "done": 4, "total": 5})

    return eq_df, fills_df, trades_df, metrics


def main() -> int:
    _ensure_repo_on_syspath()

    ap = argparse.ArgumentParser()
    ap.add_argument("--from-run", required=True, type=str)
    ap.add_argument("--config-id", required=True, type=str)
    ap.add_argument("--out-dir", default="", type=str)
    ap.add_argument("--seed", default=1, type=int)
    ap.add_argument("--starting-equity", default=10_000.0, type=float)
    ap.add_argument("--bars-per-day", default=1, type=int)
    ap.add_argument("--progress-file", default="", type=str)
    args = ap.parse_args()

    run_dir = Path(args.from_run).resolve()
    cfg_id = args.config_id.strip()

    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"--from-run must be a run directory: {run_dir}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (run_dir / "replay_cache" / cfg_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = run_dir / "batch_meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = _read_json(meta_path)
        except Exception:
            meta = {}
    else:
        # Non-fatal: replay may still work if configs_resolved carries enough info,
        # but data path will likely be missing.
        meta = {}

    data_path = _best_data_path(meta)
    df = _load_df(data_path)

    cfg_obj_raw = _load_cfg_obj_raw(run_dir, cfg_id, meta)

    # Parse/validate using whatever engine exposes publicly.
    cfg_obj: Any = cfg_obj_raw
    try:
        from engine.batch import parse_strategy_config  # type: ignore
        cfg_obj = parse_strategy_config(cfg_obj_raw)
    except Exception:
        cfg_obj = cfg_obj_raw

    progress_path = Path(args.progress_file).resolve() if args.progress_file else (out_dir / "progress.jsonl")

    _progress_write(progress_path, {"stage": "replay", "phase": "init", "done": 0, "total": 5, "out_dir": str(out_dir), "config_id": cfg_id})

    equity, fills_df, trades_df, metrics = _run_replay_once(
        df=df,
        cfg_obj_raw=cfg_obj_raw,
        meta=meta,
        seed=int(args.seed),
        starting_equity=float(args.starting_equity),
        progress_path=progress_path,
    )

    _progress_write(progress_path, {"stage": "replay", "phase": "write", "done": 4, "total": 5})

    # Write artifacts
    equity.to_csv(out_dir / "equity_curve.csv", index=False)
    trades_df.to_csv(out_dir / "trades.csv", index=False)
    fills_df.to_csv(out_dir / "fills.csv", index=False)

    # Helpful metadata
    info = {
        "config_id": cfg_id,
        "from_run": str(run_dir),
        "data_path": str(data_path),
        "seed": int(args.seed),
        "starting_equity": float(args.starting_equity),
        "bars_per_day": int(args.bars_per_day),
        "metrics": _to_jsonable(metrics),
        "cfg_raw": cfg_obj_raw,
        "written_at": time.time(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    _progress_write(progress_path, {"stage": "replay", "phase": "done", "done": 5, "total": 5, "out_dir": str(out_dir)})

    print(f"Wrote replay artifacts to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
