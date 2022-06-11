"""A rough sets toolkit for Python.

:mod:`skrough` is a Python package that provides rough sets method for experimenting
with data.
"""

from . import (
    chaos_measures,
    chaos_score,
    checks,
    dataprep,
    distributions,
    feature_importance,
    homogeneity,
    instances,
    permutations,
    predict,
    rough,
    structs,
    typing_utils,
    unique,
    weights,
)

__all__ = [
    "chaos_measures",
    "chaos_score",
    "checks",
    "dataprep",
    "distributions",
    "feature_importance",
    "homogeneity",
    "instances",
    "permutations",
    "predict",
    "rough",
    "structs",
    "typing_utils",
    "unique",
    "weights",
]
__version__ = "0.1.0"
