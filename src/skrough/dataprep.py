"""Data preparation functions.

The :mod:`skrough.dataprep` module delivers helper functions to prepare data to the form
required by other methods and algorithms.
"""

from __future__ import annotations

import logging
from typing import Literal, overload

import numpy as np
import pandas as pd

import skrough.typing as rght
from skrough.logs import log_start_end

DEFAULT_SHUFFLED_PREFIX = "shuffled_"


logger = logging.getLogger(__name__)


@overload
def prepare_factorized_vector(
    values: np.ndarray,
    return_unique_values: Literal[False] = False,
) -> tuple[np.ndarray, int]:
    ...


@overload
def prepare_factorized_vector(
    values: np.ndarray,
    return_unique_values: Literal[True],
) -> tuple[np.ndarray, int, np.ndarray]:
    ...


# TODO: add handling also for pd.Series
@log_start_end(logger)
def prepare_factorized_vector(
    values: np.ndarray, return_unique_values: bool = False
) -> tuple[np.ndarray, int] | tuple[np.ndarray, int, np.ndarray]:
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
    # TODO: check if get_uniques_and_compacted can be used instead of pd.factorize
    factorized_values, uniques = pd.factorize(values, use_na_sentinel=False)
    count_distinct = len(uniques)
    if return_unique_values:
        return factorized_values, count_distinct, uniques
    return factorized_values, count_distinct


# TODO: add handling also for pd.DataFrame
@log_start_end(logger)
def prepare_factorized_array(
    data_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Factorize data table.

    Factorize data table and return statistics of feature domain sizes.

    Args:
        data_x: A dataset to be factorized.

    Returns:
        Result is consisted of the following elements

        - factorized data returned in a form of a 2D array
        - data feature domain sizes returned in a form of 1d array, i.e., a single value
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
    target_attr: str | int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Factorize conditional and target attrs from data frame.

    Factorize data frame and return statistics of feature domain sizes for conditional
    and target attrs.

    Args:
        df: A dataset to be factorized.
        target_attr: Identifier of the target column in the input dataset.

    Returns:
        Result is consisted of the following elements

        - factorized conditional data returned in a form of a 2D array
        - conditional data feature domain sizes returned in a form of 1D array, i.e., a
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
    # pylint: disable-next=unbalanced-tuple-unpacking
    y, y_count = prepare_factorized_vector(data_y.to_numpy())
    return x, x_counts, y, y_count


# TODO: make target_attr optional - so one can shuffle just conditional attrs without
# the need the target attr to be present
@log_start_end(logger)
def add_shuffled_attrs(
    df: pd.DataFrame,
    target_attr: str | int,
    shuffled_attrs_prefix: str = DEFAULT_SHUFFLED_PREFIX,
    seed: rght.Seed = None,
) -> pd.DataFrame:
    """Add shuffled attrs.

    Add shuffled counterpart attribute for each conditional attribute (for all but one
    distinguished target attribute) of the input dataset. A shuffled (reordered)
    attribute for a given original attribute consists of the same values but permuted in
    random order. In other words, a shuffled attribute is an attribute of the same
    empirical distribution as the original one but (possibly) uncorrelated with the
    target attribute.

    Args:
        df: Input dataset.
        target_attr: Identifier of the target column in the input dataset.
        shuffled_attrs_prefix: A prefix for shuffled attribute names.
        seed: Random seed. Defaults to :obj:`None`.

    Returns:
        A dataset with shuffled counterpart attributes added.

    Examples:
        >>> df = pd.DataFrame([[5, 3, 3],
        ...                    [9, 3, 1],
        ...                    [5, 2, 3]], columns=["a", "b", "d"])
        >>> add_shuffled_attrs(df, target_attr="d", shuffled_attrs_prefix="s_", seed=0)
           a  b  s_a  s_b  dec
        0  5  3    5    2    3
        1  9  3    5    3    1
        2  5  2    9    3    3
    """
    rng = np.random.default_rng(seed)
    data_y = df[target_attr]
    data_x = df.drop(columns=target_attr)
    data_x_shuffled = data_x.apply(rng.permutation)
    col_names = list(shuffled_attrs_prefix + data_x_shuffled.columns.astype(str))
    data_x_shuffled.columns = col_names
    result = pd.concat([data_x, data_x_shuffled, data_y], axis=1)
    return result
