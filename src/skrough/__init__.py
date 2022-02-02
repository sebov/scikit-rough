"""A rough sets toolkit for Python.

skrough is a Python module that provides low level rough sets method for
experimenting with data
"""

from . import (
    chaos_score,
    dataprep,
    distributions,
    feature_importance,
    group_index,
    measures,
    reducts,
    struct,
)

__all__ = [
    "chaos_score",
    "dataprep",
    "distributions",
    "feature_importance",
    "group_index",
    "measures",
    "reducts",
    "struct",
]
__version__ = "0.1.0"
