"""
Retrying a call (an LLM call, a batch status check): up to max_retries fresh calls after a failed
one, optionally waiting with exponential backoff in between. Each failed attempt is logged as a
warning with its reason; if every attempt fails, the last reason is raised.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

from pipeline.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class Backoff:
    """Wait initial_delay seconds before the first retry, doubling for each further one, at most max_delay."""
    initial_delay: float
    max_delay: float

    def delay(self, retry_number: int) -> float:
        """Seconds to wait before retry number retry_number (1-based)."""
        return min(self.initial_delay * 2 ** (retry_number - 1), self.max_delay)


def call_with_retries(
    call: Callable[[], T],
    max_retries: int,
    what: str,
    retry_on: type[BaseException] = Exception,
    error_type: Callable[[str], BaseException] = RuntimeError,
    backoff: Optional[Backoff] = None,
) -> T:
    """
    Returns the first successful result of call(), tried up to 1 + max_retries times (with
    backoff: waiting backoff.delay(n) before retry n). Only `retry_on` exceptions are retried;
    when the attempts run out, raises error_type("<last reason> (after N attempt(s))") from the
    last exception.
    """
    attempts = max_retries + 1
    number = 1
    while True:
        try:
            return call()
        except retry_on as e:
            retrying = number < attempts
            wait = backoff.delay(number) if retrying and backoff else 0
            logger.warning(f"{what} attempt {number}/{attempts} failed: {e}"
                           + (f"; retrying in {wait:g}s." if wait else "; retrying with a new call." if retrying else ""))
            if not retrying:
                raise error_type(f"{e} (after {attempts} attempt(s))") from e
            if wait:
                time.sleep(wait)
        number += 1
