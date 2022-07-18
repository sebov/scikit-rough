import numpy as np
import pandas as pd
import pytest

from skrough.chaos_measures import conflicts_number, entropy, gini_impurity
from skrough.chaos_score import get_chaos_score_for_data, get_chaos_score_stats
from skrough.dataprep import (
    prepare_factorized_array,
    prepare_factorized_data,
    prepare_factorized_vector,
)


@pytest.fixture(name="test_data", scope="session")
def fixture_test_data():
    df = pd.DataFrame(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
        ]
    )
    x, x_counts, y, y_count = prepare_factorized_data(df, target_attr=3)
    return x, x_counts, y, y_count


@pytest.mark.parametrize(
    "chaos_fun",
    [
        conflicts_number,
        entropy,
        gini_impurity,
    ],
)
@pytest.mark.parametrize(
    "attrs, expected_distribution",
    [
        (
            [0],
            [[1, 3]],
        ),
        (
            [1],
            [
                [1, 1],
                [0, 2],
            ],
        ),
        (
            [2],
            [
                [1, 0],
                [0, 3],
            ],
        ),
        (
            [0, 1],
            [
                [1, 1],
                [0, 2],
            ],
        ),
        (
            [0, 1, 2],
            [
                [1, 0],
                [0, 1],
                [0, 2],
            ],
        ),
        (
            None,
            [
                [1, 0],
                [0, 1],
                [0, 2],
            ],
        ),
    ],
)
def test_get_chaos_score_for_data(attrs, expected_distribution, chaos_fun, test_data):
    result = get_chaos_score_for_data(*test_data, chaos_fun=chaos_fun, attrs=attrs)
    expected_distribution = np.asarray(expected_distribution)
    expected = chaos_fun(expected_distribution, expected_distribution.sum())
    assert result == expected


@pytest.mark.parametrize(
    "chaos_fun",
    [
        conflicts_number,
        entropy,
        gini_impurity,
    ],
)
@pytest.mark.parametrize(
    "x, y",
    [
        (
            [
                [0, 0],
                [0, 1],
            ],
            [0, 1],
        ),
        (
            [
                [0, 0],
                [0, 1],
            ],
            [0, 0],
        ),
        (
            [
                [0, 0, 0, 0],
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [0, 1, 1, 1],
            ],
            [0, 0, 0, 0],
        ),
        (
            [
                [0, 0, 0, 0],
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [0, 1, 1, 1],
            ],
            [0, 0, 1, 1],
        ),
        (
            [
                [0, 0, 0, 0],
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [0, 1, 1, 1],
            ],
            [0, 1, 2, 3],
        ),
    ],
)
def test_get_chaos_score_stats(x, y, chaos_fun):
    x, x_counts = prepare_factorized_array(np.asarray(x))
    y, y_count = prepare_factorized_vector(np.asarray(y))
    result = get_chaos_score_stats(
        x,
        x_counts,
        y,
        y_count,
        chaos_fun,
        increment_attrs=None,
        epsilon=None,
    )
    expected_base = get_chaos_score_for_data(
        x=x, x_counts=x_counts, y=y, y_count=y_count, chaos_fun=chaos_fun, attrs=[]
    )
    expected_total = get_chaos_score_for_data(
        x=x, x_counts=x_counts, y=y, y_count=y_count, chaos_fun=chaos_fun, attrs=None
    )
    assert result.base == expected_base
    assert result.total == expected_total
    assert result.for_increment_attrs is None
    assert result.approx_threshold is None


@pytest.mark.parametrize(
    "epsilon",
    [0, 0.1, 0.25, 0.5, 1],
)
@pytest.mark.parametrize(
    "chaos_fun",
    [
        conflicts_number,
        entropy,
        gini_impurity,
    ],
)
@pytest.mark.parametrize(
    "x, y",
    [
        (
            [
                [0, 0, 0, 0],
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [0, 1, 1, 1],
            ],
            [0, 0, 0, 0],
        ),
        (
            [
                [0, 0, 0, 0],
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [0, 1, 1, 1],
            ],
            [0, 0, 1, 1],
        ),
        (
            [
                [0, 0, 0, 0],
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [0, 1, 1, 1],
            ],
            [0, 1, 2, 3],
        ),
    ],
)
def test_get_chaos_score_stats_epsilon(x, y, chaos_fun, epsilon):
    x, x_counts = prepare_factorized_array(np.asarray(x))
    y, y_count = prepare_factorized_vector(np.asarray(y))
    result = get_chaos_score_stats(
        x,
        x_counts,
        y,
        y_count,
        chaos_fun,
        increment_attrs=None,
        epsilon=epsilon,
    )
    expected_base = get_chaos_score_for_data(
        x=x, x_counts=x_counts, y=y, y_count=y_count, chaos_fun=chaos_fun, attrs=[]
    )
    expected_total = get_chaos_score_for_data(
        x=x, x_counts=x_counts, y=y, y_count=y_count, chaos_fun=chaos_fun, attrs=None
    )

    assert result.base == expected_base
    assert result.total == expected_total
    assert result.for_increment_attrs is None
    assert result.approx_threshold is not None

    base_total_delta = result.base - result.total
    if np.isclose(base_total_delta, 0):
        assert np.isclose(result.total, result.approx_threshold)
    else:
        approx_total_delta = result.approx_threshold - result.total
        alternate_epsilon = approx_total_delta / base_total_delta
        assert np.isclose(alternate_epsilon, epsilon)


@pytest.mark.parametrize(
    "chaos_fun",
    [
        conflicts_number,
        entropy,
        gini_impurity,
    ],
)
@pytest.mark.parametrize(
    "x, y",
    [
        (
            np.empty(shape=(0, 0)),
            [],
        ),
        (
            np.empty(shape=(0, 1)),
            [],
        ),
        (
            np.empty(shape=(1, 0)),
            [0],
        ),
        (
            np.empty(shape=(1, 0)),
            [10],
        ),
        (
            [
                [0, 0, 0, 0],
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [0, 1, 1, 1],
            ],
            [0, 0, 0, 0],
        ),
        (
            [
                [0, 0, 0, 0],
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [0, 1, 1, 1],
            ],
            [0, 0, 1, 1],
        ),
        (
            [
                [0, 0, 0, 0],
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [0, 1, 1, 1],
            ],
            [0, 1, 0, 1],
        ),
        (
            [
                [0, 0, 0, 0],
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [0, 1, 1, 1],
            ],
            [0, 1, 2, 3],
        ),
    ],
)
def test_get_chaos_score_stats_increment(x, y, chaos_fun):
    x, x_counts = prepare_factorized_array(np.asarray(x))
    y, y_count = prepare_factorized_vector(np.asarray(y))
    increment_attrs = []
    for i in range(x.shape[1]):
        increment_attrs.append(list(range(i)))

    result = get_chaos_score_stats(
        x,
        x_counts,
        y,
        y_count,
        chaos_fun,
        increment_attrs=increment_attrs,
        epsilon=None,
    )

    expected_base = get_chaos_score_for_data(
        x=x, x_counts=x_counts, y=y, y_count=y_count, chaos_fun=chaos_fun, attrs=[]
    )
    expected_total = get_chaos_score_for_data(
        x=x, x_counts=x_counts, y=y, y_count=y_count, chaos_fun=chaos_fun, attrs=None
    )
    expected_for_increment_attrs = []
    for increment_attrs_element in increment_attrs:
        expected_increment_chaos_score = get_chaos_score_for_data(
            x=x,
            x_counts=x_counts,
            y=y,
            y_count=y_count,
            chaos_fun=chaos_fun,
            attrs=increment_attrs_element,
        )
        expected_for_increment_attrs.append(expected_increment_chaos_score)

    assert result.base == expected_base
    assert result.total == expected_total
    assert result.approx_threshold is None
    assert result.for_increment_attrs is not None
    assert np.allclose(result.for_increment_attrs, expected_for_increment_attrs)
