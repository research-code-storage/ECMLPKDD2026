from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def is_retryable_exception(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)

    if isinstance(status_code, int) and (status_code == 408 or status_code == 409 or status_code == 429 or status_code >= 500):
        return True

    exception_name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    retryable_keywords = (
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "rate limit",
        "ratelimit",
        "try again",
        "overloaded",
        "internal server error",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
    )
    return any(keyword in exception_name or keyword in message for keyword in retryable_keywords)


def call_with_retry(
    operation: Callable[[], T],
    *,
    max_retries: int,
    delays: tuple[int, ...] = (2, 5),
) -> tuple[T, int]:
    attempt = 0
    while True:
        try:
            return operation(), attempt
        except Exception as exc:
            if attempt >= max_retries or not is_retryable_exception(exc):
                raise
            delay = delays[min(attempt, len(delays) - 1)]
            time.sleep(delay)
            attempt += 1
