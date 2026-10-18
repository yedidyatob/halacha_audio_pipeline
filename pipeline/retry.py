"""
Retrying an LLM call: one first attempt (by default the call itself, or e.g. a batch job's
result), then up to max_retries fresh synchronous calls. Each failed attempt is logged as a
warning with its reason; if every attempt fails, the last reason is raised.
"""
from __future__ import annotations

from typing import Callable, Optional, TypeVar

from pipeline.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def call_with_retries(
    call: Callable[[], T],
    max_retries: int,
    what: str,
    first_attempt: Optional[Callable[[], T]] = None,
    retry_on: type[BaseException] = Exception,
    error_type: Callable[[str], BaseException] = RuntimeError,
) -> T:
    """
    Returns the first successful result of first_attempt (default: call), then call() up to
    max_retries times. Only `retry_on` exceptions are retried; when the attempts run out,
    raises error_type("<last reason> (after N attempt(s))") from the last exception.
    """
    attempts = max_retries + 1
    number = 1
    while True:
        try:
            return (first_attempt or call)() if number == 1 else call()
        except retry_on as e:
            retrying = number < attempts
            logger.warning(f"{what} attempt {number}/{attempts} failed: {e}"
                           + ("; retrying with a new call." if retrying else ""))
            if not retrying:
                raise error_type(f"{e} (after {attempts} attempt(s))") from e
        number += 1
