"""A rough sets toolkit for Python.

:mod:`skrough` is a Python package that provides rough sets method for experimenting
with data.
"""

from . import (
    chaos_score,
    checks,
    containers,
    dataprep,
    distributions,
    feature_importance,
    instances,
    permutations,
    unique,
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
    "instances",
    "permutations",
    "unique",
    "weights",
]
__version__ = "0.1.0"
