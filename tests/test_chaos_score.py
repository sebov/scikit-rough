from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest

from skrough.dataprep import (
    prepare_factorized_array,
    prepare_factorized_data,
    prepare_factorized_vector,
)
from skrough.disorder_measures import conflicts_count, entropy, gini_impurity
from skrough.disorder_score import get_disorder_score_for_data, get_disorder_score_stats
from tests.helpers import generate_data


def prepare_result(x, y, disorder_fun, increment_attrs, epsilon):
    x, x_counts = prepare_factorized_array(np.asarray(x))
    y, y_count = prepare_factorized_vector(np.asarray(y))
    result = get_disorder_score_stats(
        x,
        x_counts,
        y,
        y_count,
        disorder_fun,
        increment_attrs=increment_attrs,
        epsilon=epsilon,
    )
    return x, x_counts, y, y_count, result


def assert_base_and_total(x, x_counts, y, y_count, disorder_fun, result):
    expected_base = get_disorder_score_for_data(
        x=x,
        x_counts=x_counts,
        y=y,
        y_count=y_count,
        disorder_fun=disorder_fun,
        attrs=[],
    )
    expected_total = get_disorder_score_for_data(
        x=x,
        x_counts=x_counts,
        y=y,
        y_count=y_count,
        disorder_fun=disorder_fun,
        attrs=None,
    )
    assert result.base == expected_base
    assert result.total == expected_total


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
    "disorder_fun",
    [
        conflicts_count,
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
def test_get_disorder_score_for_data(
    attrs, expected_distribution, disorder_fun, test_data
):
    result = get_disorder_score_for_data(
        *test_data, disorder_fun=disorder_fun, attrs=attrs
    )
    expected_distribution = np.asarray(expected_distribution)
    expected = disorder_fun(expected_distribution, expected_distribution.sum())
    assert result == expected


@pytest.mark.parametrize(
    "disorder_fun",
    [
        conflicts_count,
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
def test_get_disorder_score_stats(x, y, disorder_fun):
    x, x_counts, y, y_count, result = prepare_result(
        x=x,
        y=y,
        disorder_fun=disorder_fun,
        increment_attrs=None,
        epsilon=None,
    )

    assert_base_and_total(
        x=x,
        x_counts=x_counts,
        y=y,
        y_count=y_count,
        disorder_fun=disorder_fun,
        result=result,
    )

    assert result.for_increment_attrs is None
    assert result.approx_threshold is None


@pytest.mark.parametrize(
    "epsilon",
    [0, 0.1, 0.25, 0.5, 1],
)
@pytest.mark.parametrize(
    "disorder_fun",
    [
        conflicts_count,
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
def test_get_disorder_score_stats_epsilon(x, y, disorder_fun, epsilon):
    x, x_counts, y, y_count, result = prepare_result(
        x=x,
        y=y,
        disorder_fun=disorder_fun,
        increment_attrs=None,
        epsilon=epsilon,
    )

    assert_base_and_total(
        x=x,
        x_counts=x_counts,
        y=y,
        y_count=y_count,
        disorder_fun=disorder_fun,
        result=result,
    )

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
    "epsilon",
    [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0],
)
def test_get_disorder_score_stats_epsilon_in_range(epsilon):
    with does_not_raise():
        prepare_result(
            x=[
                [0, 0],
                [0, 1],
            ],
            y=[0, 1],
            disorder_fun=gini_impurity,
            increment_attrs=None,
            epsilon=epsilon,
        )


@pytest.mark.parametrize(
    "epsilon",
    [-1, -0.1, -0.00001, 1.0001, 1.1, 2],
)
def test_get_disorder_score_stats_epsilon_out_of_range(epsilon):
    with pytest.raises(
        ValueError,
        match="Epsilon value should be a number between 0.0 and 1.0 inclusive",
    ):
        prepare_result(
            x=[
                [0, 0],
                [0, 1],
            ],
            y=[0, 1],
            disorder_fun=gini_impurity,
            increment_attrs=None,
            epsilon=epsilon,
        )


@pytest.mark.parametrize(
    "disorder_fun",
    [
        conflicts_count,
        entropy,
        gini_impurity,
    ],
)
@pytest.mark.parametrize(
    "x, y",
    [
        (
            generate_data(size=(0, 0)),
            [],
        ),
        (
            generate_data(size=(0, 1)),
            [],
        ),
        (
            generate_data(size=(1, 0)),
            [0],
        ),
        (
            generate_data(size=(1, 0)),
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
def test_get_disorder_score_stats_increment(x, y, disorder_fun):
    increment_attrs = []
    for i in range(np.asarray(x).shape[1]):
        increment_attrs.append(list(range(i)))
    x, x_counts, y, y_count, result = prepare_result(
        x=x,
        y=y,
        disorder_fun=disorder_fun,
        increment_attrs=increment_attrs,
        epsilon=None,
    )

    assert_base_and_total(
        x=x,
        x_counts=x_counts,
        y=y,
        y_count=y_count,
        disorder_fun=disorder_fun,
        result=result,
    )

    expected_for_increment_attrs = []
    for increment_attrs_element in increment_attrs:
        expected_increment_disorder_score = get_disorder_score_for_data(
            x=x,
            x_counts=x_counts,
            y=y,
            y_count=y_count,
            disorder_fun=disorder_fun,
            attrs=increment_attrs_element,
        )
        expected_for_increment_attrs.append(expected_increment_disorder_score)

    assert result.approx_threshold is None
    assert result.for_increment_attrs is not None
    assert np.allclose(result.for_increment_attrs, expected_for_increment_attrs)
