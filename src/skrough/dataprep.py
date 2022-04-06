"""Data preparation functions.

The :mod:`dataprep` delivers helper functions to prepare data to the form
required by other methods and algorithms.
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd
import sklearn.utils

import skrough.typing as rght


def _prepare_values(values: np.ndarray):
    """Prepare enumerated values along with a number of distinct values."""
    factorized_values, uniques = pd.factorize(values, na_sentinel=None)  # type: ignore
    count_distinct = len(uniques)
    return factorized_values, count_distinct


def prepare_factorized_data(
    df: pd.DataFrame,
    target_column: Union[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Factorize data table.

    Factorize data table and return statistics of feature domain sizes.

    Args:
        df: A dataset to be factorized.
        target_column: Index or column label.

    Returns:
        Result is consisted of the following elements

        - factorized conditional data
        - conditional data feature domain sizes
        - factorized target data
        - target feature domain size
    """
    data_y = df[target_column]
    data_x = df.drop(columns=target_column)
    res1, res2 = map(
        list,
        zip(*(_prepare_values(values) for _, values in data_x.items())),
    )
    x = np.column_stack(res1)
    x_count_distinct = np.asarray(res2)
    y, y_count_distinct = _prepare_values(data_y)
    x, y = sklearn.utils.check_X_y(x, y, multi_output=False)
    return x, x_count_distinct, y, y_count_distinct


def add_shadow_attrs(
    df: pd.DataFrame,
    target_column: Union[str, int],
    shadow_attrs_prefix: str,
    seed: rght.Seed = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data_y = df[target_column]
    data_x = df.drop(columns=target_column)
    data_x_shadow = data_x.apply(lambda col: rng.permutation(col))
    col_names = list(shadow_attrs_prefix + data_x_shadow.columns.astype(str))
    data_x_shadow.columns = col_names
    result = pd.concat([data_x, data_x_shadow, data_y], axis=1)
    return result
