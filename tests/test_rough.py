import itertools
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

from skrough.dataprep import prepare_factorized_data
from skrough.rough import get_approximations, get_gamma_value, get_positive_region

from . import datasets

EMPTY_DF_TARGET_NAME = "dec"


@pytest.fixture(scope="session")
def empty_df():
    return pd.DataFrame([], columns=["a", "b", "c", EMPTY_DF_TARGET_NAME])


@pytest.fixture(scope="session")
def empty_df_target_name():
    return EMPTY_DF_TARGET_NAME


@pytest.fixture(scope="session")
def golf_dataset_prep_one_decision(golf_dataset, golf_dataset_target_attr):
    df = golf_dataset.copy()
    df[golf_dataset_target_attr] = "no"
    return df


@pytest.fixture(scope="session")
def factorized_golf_one_decision():
    df = datasets.golf_dataset()
    df["Play"] = "no"
    return prepare_factorized_data(df, "Play")


# https://docs.python.org/3/library/itertools.html#itertools-recipes
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def pos_alternative_impl(df: pd.DataFrame, dec: str, attrs: Sequence[int]):
    if not attrs:
        if len(df.loc[:, dec].unique()) == 1:
            result = df.index[:]
        else:
            result = df.index[[]]
    else:
        col_names = list(df.columns[attrs])
        result = df.groupby(col_names)[dec].filter(lambda x: x.nunique() == 1).index
    return result.to_list()


def gamma_alternative_impl(df: pd.DataFrame, dec: str, attrs: Sequence[int]):
    if len(df) == 0:
        return 1
    return len(pos_alternative_impl(df, dec, attrs)) / len(df)


def approx_alternative_impl(
    df: pd.DataFrame,
    objs: Sequence[int],
    attrs: Sequence[int],
):
    if attrs:
        col_names = list(df.columns[attrs])
        lower = df.groupby(col_names).filter(lambda x: all(x.index.isin(objs))).index
        upper = df.groupby(col_names).filter(lambda x: any(x.index.isin(objs))).index
        lower = lower.to_list()
        upper = upper.to_list()
    else:
        lower = df.index.to_list() if all(df.index.isin(objs)) else []
        upper = df.index.to_list() if any(df.index.isin(objs)) else []
    return lower, upper


def run_compare_pos(
    factorized_data: Tuple[np.ndarray, np.ndarray, np.ndarray, int],
    df: pd.DataFrame,
    dec: str,
    attrs: Sequence[int],
):
    attrs = list(attrs)
    assert get_positive_region(*factorized_data, attrs) == pos_alternative_impl(
        df, dec, attrs
    )


def run_compare_gamma(
    factorized_data: Tuple[np.ndarray, np.ndarray, np.ndarray, int],
    df: pd.DataFrame,
    dec: str,
    attrs: Sequence[int],
):
    attrs = list(attrs)
    assert get_gamma_value(*factorized_data, attrs) == gamma_alternative_impl(
        df, dec, attrs
    )


def run_compare_approx(
    factorized_data: Tuple[np.ndarray, np.ndarray, np.ndarray, int],
    df: pd.DataFrame,
    dec: str,
    objs: Sequence[int],
    attrs: Sequence[int],
):
    x, x_counts, y, y_count = factorized_data
    assert get_approximations(x, x_counts, objs, attrs) == approx_alternative_impl(
        df, objs, attrs
    )


@pytest.mark.parametrize("attrs", powerset([0, 1, 2, 3]))
def test_pos(attrs, golf_dataset_prep, golf_dataset, golf_dataset_target_attr):
    run_compare_pos(golf_dataset_prep, golf_dataset, golf_dataset_target_attr, attrs)


@pytest.mark.parametrize("attrs", powerset([0, 1, 2, 3]))
def test_pos_one_decision(
    attrs,
    factorized_golf_one_decision,
    golf_dataset_prep_one_decision,
    golf_dataset_target_attr,
):
    run_compare_pos(
        factorized_golf_one_decision,
        golf_dataset_prep_one_decision,
        golf_dataset_target_attr,
        attrs,
    )


def test_pos_empty_df(empty_df, empty_df_target_name):
    factorized_data = prepare_factorized_data(empty_df, empty_df_target_name)
    run_compare_pos(factorized_data, empty_df, empty_df_target_name, [])


@pytest.mark.parametrize("attrs", powerset([0, 1, 2, 3]))
def test_gamma(attrs, golf_dataset_prep, golf_dataset, golf_dataset_target_attr):
    run_compare_gamma(golf_dataset_prep, golf_dataset, golf_dataset_target_attr, attrs)


@pytest.mark.parametrize("attrs", powerset([0, 1, 2, 3]))
def test_gamma_one_decision(
    attrs,
    factorized_golf_one_decision,
    golf_dataset_prep_one_decision,
    golf_dataset_target_attr,
):
    run_compare_gamma(
        factorized_golf_one_decision,
        golf_dataset_prep_one_decision,
        golf_dataset_target_attr,
        attrs,
    )


def test_gamma_empty_df(empty_df, empty_df_target_name):
    factorized_data = prepare_factorized_data(empty_df, empty_df_target_name)
    run_compare_gamma(factorized_data, empty_df, empty_df_target_name, [])


@pytest.mark.parametrize(
    "attrs",
    powerset([0, 1, 2, 3]),
)
@pytest.mark.parametrize(
    "objs",
    [
        [],
        [0],
        [0, 1, 3],
        [1, 13],
        [5, 12, 13],
        [1, 6, 9, 10],
        [3, 4, 5, 6, 7, 8],
        [5, 6, 7],
        [0, 2, 4, 6, 8, 10, 12],
        [1, 3, 5, 7, 9, 11, 13],
        [10, 11, 13],
    ],
)
def test_approximations(
    objs,
    attrs,
    golf_dataset_prep,
    golf_dataset,
    golf_dataset_target_attr,
):
    run_compare_approx(
        golf_dataset_prep, golf_dataset, golf_dataset_target_attr, objs, list(attrs)
    )
