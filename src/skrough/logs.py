"""Logging utils."""

import functools
import logging


def log_call(fun):
    """Decorator that logs entry and exit of a function.

    Uses the logger of the module where the decorated function is defined.
    Logs ``enter <func_name>`` and ``exit <func_name>`` at DEBUG level.

    Args:
        fun: The function to decorate.

    Returns:
        Wrapped function that logs entry and exit.
    """

    @functools.wraps(fun)
    def decorated(*args, **kwargs):
        logger = logging.getLogger(fun.__module__)
        logger.debug("enter %s", fun.__name__)
        result = fun(*args, **kwargs)
        logger.debug("exit %s", fun.__name__)
        return result

    return decorated
