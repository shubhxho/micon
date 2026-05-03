"""Centralized loguru-backed logging for the micom package.

A single place that:

1. Configures one ``loguru`` sink for the whole process (stderr by default,
   coloured, with module + line metadata).
2. Bridges any stdlib ``logging`` records into the same sink so libraries
   that still use ``logging.getLogger`` (pydicom, modal, hf_hub, monai, ...)
   land in the same stream — no duplicate output, no lost messages.
3. Exposes a tiny ``get_logger(name)`` helper that returns a ``loguru``
   logger bound to that module name. Existing modules can swap
   ``logger = logging.getLogger(__name__)`` for
   ``logger = get_logger(__name__)`` with no other changes — the call
   surface (``.info``, ``.debug``, ``.warning``, ``.exception``,
   ``.bind(**ctx)``, lazy ``%``-style format, ``extra=``) is compatible.

The configuration is idempotent and cheap: importing this module from
multiple places never installs duplicate sinks.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from loguru import logger as _loguru_logger

__all__ = ["configure", "get_logger", "logger"]


_CONFIGURED = False


class _InterceptHandler(logging.Handler):
    """Forward stdlib ``logging`` records into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = _loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        _loguru_logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure(
    *,
    level: str | None = None,
    serialize: bool | None = None,
    sink: Any = sys.stderr,
    force: bool = False,
) -> None:
    """Install the loguru sink and stdlib bridge.

    First call (auto-fired at import) wins by default. Pass ``force=True`` to
    re-configure (e.g. when a CLI ``--verbose`` flag flips the level).

    Parameters
    ----------
    level:
        Minimum log level. Defaults to ``LOGURU_LEVEL`` env var or ``INFO``.
    serialize:
        Emit JSON records (good for Modal / structured ingestion).
        Defaults to ``LOGURU_JSON`` env var truthiness.
    sink:
        Loguru sink (file path, stream, callable). Defaults to ``sys.stderr``.
    force:
        If True, re-install the sink even if ``configure`` ran earlier.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    resolved_level = (level or os.environ.get("LOGURU_LEVEL") or "INFO").upper()
    resolved_serialize = (
        serialize
        if serialize is not None
        else os.environ.get("LOGURU_JSON", "").lower() in {"1", "true", "yes"}
    )

    _loguru_logger.remove()
    _loguru_logger.add(
        sink,
        level=resolved_level,
        serialize=resolved_serialize,
        backtrace=False,
        diagnose=False,
        enqueue=False,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
            "<level>{level: <8}</level> "
            "<cyan>{extra[name]}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for name in ("urllib3", "asyncio", "modal", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> Any:
    """Return a loguru logger bound to *name* (drop-in for ``logging.getLogger``)."""
    if not _CONFIGURED:
        configure()
    return _loguru_logger.bind(name=name)


configure()
logger = _loguru_logger.bind(name="micom")
