"""A rough sets toolkit for Python.

skrough is a Python module that provides low level rough sets method for
experimenting with data
"""

from . import (
    chaos_score,
    containers,
    dataprep,
    distributions,
    feature_importance,
    group_index,
    measures,
    reducts,
)

__all__ = [
    "chaos_score",
    "containers",
    "dataprep",
    "distributions",
    "feature_importance",
    "group_index",
    "measures",
    "reducts",
]
__version__ = "0.1.0"
