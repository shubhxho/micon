"""Pipeline package -- each stage is a separate module.

Re-exports the most common runtime helpers so callers can write::

    from src.pipeline import (
        mark_started, mark_finished, is_done, summarize,
        log, log_error,
        RunReport, plan,
    )

instead of reaching into the leaf modules every time.
"""

from .discover import discover_dcm_folders
from .log import log, log_error
from .plan import plan
from .run import run_pipeline
from .run_report import RunReport
from .sentinels import is_done, mark_finished, mark_started, summarize

__all__ = [
    "RunReport",
    "discover_dcm_folders",
    "is_done",
    "log",
    "log_error",
    "mark_finished",
    "mark_started",
    "plan",
    "run_pipeline",
    "summarize",
]
