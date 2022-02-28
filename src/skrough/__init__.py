"""A rough sets toolkit for Python.

skrough is a Python module that provides low level rough sets method for
experimenting with data
"""

from . import (
    chaos_score,
    checks,
    containers,
    dataprep,
    distributions,
    feature_importance,
    group_index,
    instances,
    permutations,
    reducts,
    weights,
)
from .chaos_measures import chaos_measures

__all__ = [
    "chaos_measures",
    "chaos_score",
    "checks",
    "containers",
    "dataprep",
    "distributions",
    "feature_importance",
    "group_index",
    "instances",
    "permutations",
    "reducts",
    "weights",
]
__version__ = "0.1.0"
