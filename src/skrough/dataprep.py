"""Data preparation functions.

The :mod:`dataprep` delivers helper functions to prepare data to the form
required by other algorithms.
"""

from typing import Tuple

import numpy as np
import pandas as pd
import sklearn.utils


def _prepare_values(values):
    """Factorize values."""
    factorized_values, uniques = pd.factorize(values, na_sentinel=None)  # type: ignore
    uniques = len(uniques)
    return factorized_values, uniques


def prepare_df(
    df: pd.DataFrame, target_column
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Factorize data table.

    Factorize data table and return statistics of feature domain sizes.

    Parameters
    ----------
    df : pd.DataFrame
        A dataset to be factorize
    target_column : single label
        Index or column label

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, int]
        A tuple consisted of:
        - factorized conditional data
        - conditional data feature domain sizes
        - factorized target data
        - target feature domain size
    """
    y = df[target_column]
    x = df.drop(columns=target_column)
    data = x.apply(_prepare_values, 0)
    x = np.vstack(data.values[0]).T  # type: ignore
    x_count_distinct = data.values[1].astype(int)
    y, y_count_distinct = _prepare_values(y)
    x, y = sklearn.utils.check_X_y(x, y, multi_output=False)
    return x, x_count_distinct, y, y_count_distinct
