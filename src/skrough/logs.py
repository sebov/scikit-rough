import contextlib
import logging
from typing import Optional


class log_start_end(contextlib.ContextDecorator):
    """Create start-end logging decorator.

    Returned decorator, when applied, logs start and end of the wrapped function
    using specified logger instance and the specified log level.

    Args:
        logger: Logger instance to be used.
        name: Name to be used in messages. If equals to None, the decorated function
            name will be used. Defaults to None.
        level: Logging level to be used. Defaults to logging.DEBUG.
    """

    def __init__(
        self,
        logger: logging.Logger,
        name: Optional[str] = None,
        level: int = logging.DEBUG,
    ):
        super().__init__()
        self.logger = logger
        self.name = name
        self.level = level

    def __enter__(self):
        self.logger.log(self.level, "enter %s", self.name)
        return self

    def __exit__(self, *exc):
        self.logger.log(self.level, "exit %s", self.name)
        return False

    def __call__(self, fun):
        self.name = self.name or fun.__name__
        return super().__call__(fun)
