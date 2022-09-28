"""Data preparation functions.

The :mod:`skrough.dataprep` module delivers helper functions to prepare data to the form
required by other methods and algorithms.
"""

import logging
from typing import Tuple, Union

import numpy as np
import pandas as pd

import skrough.typing as rght
from skrough.logs import log_start_end

logger = logging.getLogger(__name__)


@log_start_end(logger)
def prepare_factorized_vector(values: np.ndarray) -> Tuple[np.ndarray, int]:
    """Factorize values.

    Prepare enumerated values along with a number of distinct values.

    Args:
        values: A 1d array to be factorized.

    Returns:
        Result is consisted of the following elements

        - factorized data returned in form of 1d array
        - feature domain size

    Examples:
        >>> ar = np.array([3, 4, 3, 3, 2])
        >>> prepare_factorized_vector(ar)
        (array([0, 1, 0, 0, 2]), 3)
    """
    factorized_values, uniques = pd.factorize(values, use_na_sentinel=False)
    count_distinct = len(uniques)
    return factorized_values, count_distinct


@log_start_end(logger)
def prepare_factorized_array(
    data_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Factorize data table.

    Factorize data table and return statistics of feature domain sizes.

    Args:
        data_x: A dataset to be factorized.

    Returns:
        Result is consisted of the following elements

        - factorized data returned in form of a 2d array
        - data feature domain sizes returned in for of 1d array, i.e., a single value
          (domain size) returned for each column

    Examples:
        >>> ar = np.array([[5, 3],
        ...                [9, 3],
        ...                [5, 2]])
        >>> prepare_factorized_array(ar)
        (array([[0, 0],
                [1, 0],
                [0, 1]]),
        array([2, 2]))
    """
    if data_x.size == 0:
        return data_x, np.zeros(data_x.shape[1])
    factorized = [
        prepare_factorized_vector(data_x[:, i]) for i in range(data_x.shape[1])
    ]
    res1, res2 = zip(*factorized)
    x: np.ndarray = np.column_stack(res1)
    x_counts = np.array(res2)
    return x, x_counts


@log_start_end(logger)
def prepare_factorized_data(
    df: pd.DataFrame,
    target_attr: Union[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Factorize conditional and target attrs from data frame.

    Factorize data frame and return statistics of feature domain sizes for conditional
    and target attrs.

    Args:
        df: A dataset to be factorized.
        target_attr: Identifier of the target column in the input dataset.

    Returns:
        Result is consisted of the following elements

        - factorized conditional data returned in form of a 2d array
        - conditional data feature domain sizes returned in for of 1d array, i.e., a
          single value (domain size) returned for each column
        - factorized target data returned in form of 1d array
        - target feature domain size

    Examples:
        >>> df = pd.DataFrame([[5, 3, 3],
        ...                    [9, 3, 1],
        ...                    [5, 2, 3]], columns=["a", "b", "dec"])
        >>> prepare_factorized_data(df, target_attr="dec")
        (array([[0, 0],
                [1, 0],
                [0, 1]]),
        array([2, 2]),
        array([0, 1, 0]),
        2)
    """
    data_y = df[target_attr]
    data_x = df.drop(columns=target_attr)
    x, x_counts = prepare_factorized_array(data_x.to_numpy())
    y, y_count = prepare_factorized_vector(data_y.to_numpy())
    return x, x_counts, y, y_count


@log_start_end(logger)
def add_shadow_attrs(
    df: pd.DataFrame,
    target_attr: Union[str, int],
    shadow_attrs_prefix: str = "shadow_",
    seed: rght.Seed = None,
) -> pd.DataFrame:
    """Add shadow attrs.

    Add shadow counterpart attribute for each conditional attribute (for all but one
    distinguished target attribute) of the input dataset. A shadow (reordered) attribute
    for a given original attribute consists of the same values but shuffled in random
    order. In other words, a shadow attribute is an attribute of the same empirical
    distribution as the original one but (possibly) uncorrelated with the target
    attribute.

    Args:
        df: Input dataset.
        target_attr: Identifier of the target column in the input dataset.
        shadow_attrs_prefix: A prefix for shadow attribute names.
        seed: Random seed. Defaults to ``None``.

    Returns:
        A dataset with shadow counterpart attributes added.

    Examples:
        >>> df = pd.DataFrame([[5, 3, 3],
        ...                    [9, 3, 1],
        ...                    [5, 2, 3]], columns=["a", "b", "dec"])
        >>> add_shadow_attrs(df, target_attr="dec", shadow_attrs_prefix="s_", seed=0)
           a  b  s_a  s_b  dec
        0  5  3    5    2    3
        1  9  3    5    3    1
        2  5  2    9    3    3
    """
    rng = np.random.default_rng(seed)
    data_y = df[target_attr]
    data_x = df.drop(columns=target_attr)
    data_x_shadow = data_x.apply(rng.permutation)
    col_names = list(shadow_attrs_prefix + data_x_shadow.columns.astype(str))
    data_x_shadow.columns = col_names
    result = pd.concat([data_x, data_x_shadow, data_y], axis=1)
    return result
