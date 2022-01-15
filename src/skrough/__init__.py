"""A rough sets toolkit for Python.

skrough is a Python module that provides low level rough sets method for
experimenting with data
"""

from . import (
    base,
    chaos_score,
    dataprep,
    distributions,
    feature_importance,
    group_index,
    measures,
    reducts,
)

__all__ = [
    "base",
    "chaos_score",
    "dataprep",
    "distributions",
    "feature_importance",
    "group_index",
    "measures",
    "reducts",
]
__version__ = "0.1.0"
