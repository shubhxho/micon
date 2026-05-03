"""Resilience primitives — retries with exponential back-off via ``stamina``.

Centralizes the retry policy for network-bound calls (HF Hub, OpenAI,
Modal, S3) so individual call sites don't reinvent ad-hoc ``while True``
loops. ``stamina`` builds on ``tenacity`` with structured logging and
asyncio-aware semantics.

Usage::

    from src._resilience import retry, retry_network

    @retry_network()
    def upload_file(...): ...

    @retry(on=ValueError, attempts=5, wait_initial=1.0)
    def parse_remote_json(...): ...
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import stamina

__all__ = ["retry", "retry_network"]


F = TypeVar("F", bound=Callable[..., Any])


_DEFAULT_NETWORK_EXC: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def retry(
    *,
    on: type[BaseException] | tuple[type[BaseException], ...] = Exception,
    attempts: int = 3,
    wait_initial: float = 0.5,
    wait_max: float = 30.0,
    timeout: float | None = None,
) -> Callable[[F], F]:
    """Generic retry decorator. Thin wrapper over ``stamina.retry``."""
    return stamina.retry(  # type: ignore[return-value]
        on=on,
        attempts=attempts,
        wait_initial=wait_initial,
        wait_max=wait_max,
        timeout=timeout,
    )


def retry_network(
    *,
    attempts: int = 5,
    wait_initial: float = 1.0,
    wait_max: float = 60.0,
) -> Callable[[F], F]:
    """Retry on common network/IO transient errors (5 attempts, expo back-off)."""
    return stamina.retry(  # type: ignore[return-value]
        on=_DEFAULT_NETWORK_EXC,
        attempts=attempts,
        wait_initial=wait_initial,
        wait_max=wait_max,
    )
