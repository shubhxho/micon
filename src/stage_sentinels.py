"""Stage sentinel state machine for the resume pipeline.

Persists per-stage status as JSON files under:
    <output_dir>/_pipeline_state/<stage_name>.json

Each sentinel contains:
    {stage, started_at, finished_at, status, inputs_processed, errors, metadata}

All writes are atomic (write to .tmp then os.replace).
No Modal imports.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_STATE_DIR = "_pipeline_state"
_STATUS_IN_PROGRESS = "in_progress"
_STATUS_OK = "ok"
_STATUS_FAILED = "failed"


def _sentinel_path(output_dir: str | Path, stage: str) -> Path:
    return Path(output_dir) / _STATE_DIR / f"{stage}.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def mark_started(
    output_dir: str | Path,
    stage: str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a sentinel with status=in_progress.  Returns the sentinel path."""
    path = _sentinel_path(output_dir, stage)
    state: dict[str, Any] = {
        "stage": stage,
        "started_at": _now_iso(),
        "finished_at": None,
        "status": _STATUS_IN_PROGRESS,
        "inputs_processed": 0,
        "errors": 0,
        "metadata": metadata or {},
    }
    _atomic_write(path, state)
    return path


def mark_finished(
    output_dir: str | Path,
    stage: str,
    ok: bool,
    errors: int = 0,
    inputs_processed: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Update an existing sentinel with final status + finished_at.

    Loads the existing file so started_at is preserved.  If the sentinel does
    not exist (e.g. the caller skipped mark_started) a minimal record is
    created so nothing crashes.
    """
    path = _sentinel_path(output_dir, stage)
    existing = read_state(output_dir, stage) or {
        "stage": stage,
        "started_at": None,
        "inputs_processed": 0,
        "errors": 0,
        "metadata": {},
    }

    existing["finished_at"] = _now_iso()
    existing["status"] = _STATUS_OK if ok else _STATUS_FAILED
    existing["errors"] = errors
    if inputs_processed is not None:
        existing["inputs_processed"] = inputs_processed
    if metadata:
        existing["metadata"] = {**existing.get("metadata", {}), **metadata}

    _atomic_write(path, existing)


def is_done(output_dir: str | Path, stage: str) -> bool:
    """Return True iff the sentinel exists with status=ok."""
    state = read_state(output_dir, stage)
    return state is not None and state.get("status") == _STATUS_OK


def read_state(output_dir: str | Path, stage: str) -> dict | None:
    """Return the parsed sentinel dict, or None if it does not exist / is corrupt."""
    path = _sentinel_path(output_dir, stage)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def summarize(output_dir: str | Path) -> dict[str, dict]:
    """Return a mapping of stage_name -> state for all sentinels found."""
    state_dir = Path(output_dir) / _STATE_DIR
    result: dict[str, dict] = {}
    if not state_dir.is_dir():
        return result
    for sentinel in sorted(state_dir.glob("*.json")):
        stage = sentinel.stem
        state = read_state(output_dir, stage)
        if state is not None:
            result[stage] = state
    return result
