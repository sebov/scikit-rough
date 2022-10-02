"""Exceptions related to algorithms."""


class LoopBreak(Exception):
    """A class used to represent a loop break event.

    An exception class to be used in hook functions (cf.
    :mod:`~skrough.algorithms.hooks`) to represent loop break event causing an exit from
    execution of a particular :class:`~skrough.algorithms.meta.stage.Stage`.
    """
