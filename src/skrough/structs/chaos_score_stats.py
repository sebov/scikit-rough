"""Chaos score stats structures."""

from typing import List, Optional

from attrs import define


@define
class ChaosScoreStats:
    """A class to represent a set of chaos score statistics.

    A class to represent a set of chaos score statistics, mainly used to store results
    of the :func:`~skrough.chaos_score.get_chaos_score_stats` function.
    """

    base: float
    """Base chaos score in data - when all objects are in the same group."""

    total: float
    """Total chaos score in data - when all objects are split into into equivalence
    classes using all attributes.
    """

    for_increment_attrs: Optional[List[float]] = None
    """Intermediate chaos score values - a non-increasing chaos score values sequence
    obtained for a growing subset of attributes defined in accordance with
    :func:`~skrough.chaos_score.get_chaos_score_stats` arguments semantics."""

    approx_threshold: Optional[float] = None
    """Approximation threshold - a threshold/limit value of chaos score somewhere
    between :obj:`total` (low value) and :obj:`base` (high value) which is the goal to
    be achieved in some algorithm or heuristic procedure aiming to minimize the chaos
    score value."""
