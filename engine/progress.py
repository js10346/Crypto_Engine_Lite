"""Lightweight progress telemetry writer.

Place this file at: `engine/progress.py`

Design goals:
- Append-only JSONL (safe to tail while a process is running)
- Throttled writes (won't spam disk)
- Works on Windows + multiprocessing (main process writes updates)

Event format: one JSON object per line.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    # Example: 2026-01-30T12:34:56.789Z
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class ProgressWriter:
    """Append-only JSONL progress writer with throttling."""

    path: Path
    # Minimum time between writes (seconds)
    min_interval_sec: float = 0.25
    # Force a file flush at least this often (seconds)
    flush_interval_sec: float = 1.0

    _fp: Any = field(default=None, init=False, repr=False)
    _t_last_write: float = field(default=0.0, init=False, repr=False)
    _t_last_flush: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # line-buffered text mode
        self._fp = open(self.path, "a", encoding="utf-8", buffering=1)
        now = time.time()
        self._t_last_write = 0.0
        self._t_last_flush = now

    def close(self) -> None:
        try:
            if self._fp:
                self._fp.flush()
                self._fp.close()
        except Exception:
            pass
        self._fp = None

    def write(self, stage: str, **fields: Any) -> None:
        """Write a progress event.

        The write is throttled to at most ~1/min_interval_sec.
        To force a write, pass force=True.
        """
        force = bool(fields.pop("force", False))
        now = time.time()
        if (not force) and (now - self._t_last_write) < float(self.min_interval_sec):
            return

        evt: Dict[str, Any] = {"t": _utc_now_iso(), "stage": str(stage)}
        evt.update(fields)

        try:
            self._fp.write(json.dumps(evt, ensure_ascii=False, separators=(",", ":")) + "\n")
        except Exception:
            # Best-effort. Never crash the job because telemetry failed.
            return

        self._t_last_write = now
        if (now - self._t_last_flush) >= float(self.flush_interval_sec):
            try:
                self._fp.flush()
            except Exception:
                pass
            self._t_last_flush = now

    def heartbeat(self, stage: str, message: str, **fields: Any) -> None:
        self.write(stage, message=str(message), **fields, force=True)

