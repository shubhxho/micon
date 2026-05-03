"""Structured JSON-line logger for the resume pipeline.

Writes one JSON object per line to stdout so Modal / any log aggregator can
parse pipeline events without screen-scraping.

Example output:
    {"ts": "2026-04-30T12:00:00+00:00", "stage": "quality", "event": "start", "study_id": "abc"}

No external dependencies.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(stage: str, event: str, **fields: Any) -> None:
    """Write a JSON-line log entry to stdout.

    Args:
        stage: Pipeline stage name (e.g. "quality", "annotation").
        event: Short event label (e.g. "start", "finish", "skip").
        **fields: Arbitrary key/value pairs merged into the record.
    """
    record: dict[str, Any] = {
        "ts": _now_iso(),
        "stage": stage,
        "event": event,
        **fields,
    }
    print(json.dumps(record, default=str), file=sys.stdout, flush=True)


def log_error(stage: str, event: str, exc: BaseException, **fields: Any) -> None:
    """Write a JSON-line log entry with error class and message.

    Args:
        stage: Pipeline stage name.
        event: Short event label (e.g. "error", "retry_exhausted").
        exc: The exception instance.
        **fields: Arbitrary additional context.
    """
    log(
        stage,
        event,
        error_class=type(exc).__name__,
        error_msg=str(exc),
        **fields,
    )
