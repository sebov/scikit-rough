"""Disorder score stats structures."""

from typing import List, Optional

from attrs import define


@define
class DisorderScoreStats:
    """A class to represent a set of disorder score statistics.

    A class to represent a set of disorder score statistics, mainly used to store
    results of the :func:`~skrough.disorder_score.get_disorder_score_stats` function.
    """

    base: float
    """Base disorder score in data - when all objects are in the same group."""

    total: float
    """Total disorder score in data - when all objects are split into into equivalence
    classes using all attributes.
    """

    for_increment_attrs: Optional[List[float]] = None
    """Intermediate disorder score values - a non-increasing disorder score values
    sequence obtained for a growing subset of attributes defined in accordance with
    :func:`~skrough.disorder_score.get_disorder_score_stats` arguments semantics."""

    approx_threshold: Optional[float] = None
    """Approximation threshold - a threshold/limit value of disorder score somewhere
    between :obj:`total` (low value) and :obj:`base` (high value) which is the goal to
    be achieved in some algorithm or heuristic procedure aiming to minimize the disorder
    score value."""
