"""Logging utils."""

import functools
import logging


def log_call(fun):
    """Decorator that logs enter/exit of a function using the module's logger."""

    @functools.wraps(fun)
    def decorated(*args, **kwargs):
        logger = logging.getLogger(fun.__module__)
        logger.debug("enter %s", fun.__name__)
        result = fun(*args, **kwargs)
        logger.debug("exit %s", fun.__name__)
        return result

    return decorated
