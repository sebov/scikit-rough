"""Logging utils."""

import functools
import logging
from typing import Optional


def log_start_end(
    logger: logging.Logger,
    name: Optional[str] = None,
    level: int = logging.DEBUG,
):
    def decorator(fun):
        @functools.wraps(fun)
        def decorated(*args, **kwargs):
            reported_name = name or fun.__name__
            logger.log(level, "enter %s", reported_name)
            result = fun(*args, **kwargs)
            logger.log(level, "exit %s", reported_name)
            return result

        return decorated

    return decorator
