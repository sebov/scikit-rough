import contextlib
import logging


class log_start_end(contextlib.ContextDecorator):
    """Create start-end logging decorator.

    Returned decorator when applied logs start and end of the wrapped function
    using specified logger instance with the specified log level.

    Args:
        logger: Logger instance to be used.
        level: Logging level to be used. Defaults to logging.DEBUG.
    """

    def __init__(self, logger: logging.Logger, level: int = logging.DEBUG):
        super().__init__()
        self.logger = logger
        self.level = level

    def __enter__(self):
        self.logger.log(self.level, "enter %s", self.function_name)
        return self

    def __exit__(self, *exc):
        self.logger.log(self.level, "exit %s", self.function_name)
        return False

    def __call__(self, fun):
        self.function_name = fun.__name__
        return super().__call__(fun)
