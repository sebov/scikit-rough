"""Data preparation functions.

The :mod:`dataprep` delivers helper functions to prepare data to the form
required by other methods and algorithms.
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd

import skrough.typing as rght


def prepare_factorized_values(values: np.ndarray) -> Tuple[np.ndarray, int]:
    """Prepare enumerated values along with a number of distinct values."""
    factorized_values, uniques = pd.factorize(values, na_sentinel=None)  # type: ignore
    count_distinct = len(uniques)
    return factorized_values, count_distinct


def prepare_factorized_data(
    df: pd.DataFrame,
    target_attr: Union[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Factorize data table for conditional and target attrs.

    Factorize data table and return statistics of feature domain sizes for
    conditional and target attrs.

    Args:
        df: A dataset to be factorized.
        target_attr: Index or column label.

    Returns:
        Result is consisted of the following elements

        - factorized conditional data
        - conditional data feature domain sizes
        - factorized target data
        - target feature domain size
    """
    data_y = df[target_attr]
    data_x = df.drop(columns=target_attr)
    x, x_counts = prepare_factorized_x(data_x.to_numpy())
    y, y_count = prepare_factorized_values(data_y.to_numpy())
    return x, x_counts, y, y_count


def prepare_factorized_x(
    data_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Factorize data table.

    Factorize data table and return statistics of feature domain sizes.

    Args:
        data_x: A dataset to be factorized.

    Returns:
        Result is consisted of the following elements

        - factorized conditional data
        - data feature domain sizes
    """
    factorized = [
        prepare_factorized_values(data_x[:, i]) for i in range(data_x.shape[1])
    ]
    res1, res2 = zip(*factorized)
    x: np.ndarray = np.column_stack(res1)
    x_counts = np.array(res2)
    return x, x_counts


def add_shadow_attrs(
    df: pd.DataFrame,
    target_attr: Union[str, int],
    shadow_attrs_prefix: str,
    seed: rght.Seed = None,
) -> pd.DataFrame:
    """Add shadow attrs.

    Add shadow counterpart attribute for each conditional attribute
    (for all but one distinguished target attribute) of the input dataset.
    A shadow (reordered) attribute for a given original attribute consists
    of the same values but shuffled in random order. In other words, a shadow attribute
    is an attribute of the same empirical distribution as the original one but
    uncorrelated with the target attribute.

    Args:
        df: Input dataset.
        target_attr: Identifier of the target column in the input dataset.
        shadow_attrs_prefix: A prefix to be added to shadow attributes.
        seed: Random seed. Defaults to None.

    Returns:
        A dataset with shadow counterpart attributes added.
    """
    rng = np.random.default_rng(seed)
    data_y = df[target_attr]
    data_x = df.drop(columns=target_attr)
    data_x_shadow = data_x.apply(lambda col: rng.permutation(col))
    col_names = list(shadow_attrs_prefix + data_x_shadow.columns.astype(str))
    data_x_shadow.columns = col_names
    result = pd.concat([data_x, data_x_shadow, data_y], axis=1)
    return result
