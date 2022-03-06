"""Data preparation functions.

The :mod:`dataprep` delivers helper functions to prepare data to the form
required by other algorithms.
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd
import sklearn.utils


def _prepare_values(values):
    """Factorize values."""
    factorized_values, uniques = pd.factorize(values, na_sentinel=None)  # type: ignore
    count_distinct = len(uniques)
    return factorized_values, count_distinct


def prepare_df(
    df: pd.DataFrame,
    target_column: Union[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Factorize data table.

    Factorize data table and return statistics of feature domain sizes.

    Args:
        df: A dataset to be factorized.
        target_column: Index or column label.

    Returns:
        A tuple consisted of:
        - factorized conditional data
        - conditional data feature domain sizes
        - factorized target data
        - target feature domain size
    """
    datay = df[target_column]
    datax = df.drop(columns=target_column)
    res1, res2 = map(
        list,
        zip(*(_prepare_values(values) for _, values in datax.items())),
    )
    x = np.column_stack(res1)
    x_count_distinct = np.asarray(res2)
    y, y_count_distinct = _prepare_values(datay)
    x, y = sklearn.utils.check_X_y(x, y, multi_output=False)
    return x, x_count_distinct, y, y_count_distinct
