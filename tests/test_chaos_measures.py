"""Test reducts"""

import numpy as np
import pytest

import skrough.typing as rght
from skrough.disorder_measures.disorder_measures import (
    conflicts_count,
    entropy,
    gini_impurity,
)


def gini_impurity_alternative_impl(
    distribution: np.ndarray,
) -> float:
    all_sum = distribution.sum()
    if all_sum == 0:
        return 0
    result = 0.0
    for i in range(distribution.shape[0]):
        row_sum = distribution[i, :].sum()
        if row_sum == 0:
            continue
        row_result = 0.0
        for j in range(distribution.shape[1]):
            value = distribution[i, j]
            prob = value / row_sum
            row_result += prob * (1 - prob)
        result += (row_sum / all_sum) * row_result
    return result


def entropy_alternative_impl(
    distribution: np.ndarray,
) -> float:
    all_sum = distribution.sum()
    if all_sum == 0:
        return 0
    result = 0.0
    for i in range(distribution.shape[0]):
        row_sum = distribution[i, :].sum()
        if row_sum == 0:
            continue
        row_result = 0.0
        for j in range(distribution.shape[1]):
            value = distribution[i, j]
            if value > 0:
                prob = value / row_sum
                row_result -= prob * np.log2(prob)
        result += (row_sum / all_sum) * row_result
    return result


def conflicts_count_alternative_impl(
    distribution: np.ndarray,
) -> float:
    all_sum = distribution.sum()
    if all_sum == 0:
        return 0
    result = 0.0
    for i in range(distribution.shape[0]):
        row_sum = distribution[i, :].sum()
        if row_sum == 0:
            continue
        row_result = 0.0
        for j in range(distribution.shape[1]):
            value = distribution[i, j]
            row_result += value * (row_sum - value)
        result += row_result
    return result / 2


def run_compare_measure(
    distribution,
    measure: rght.DisorderMeasure,
    measure_alternative_impl,
):
    distribution = np.asarray(distribution)
    n_objs = distribution.sum()
    result = measure(distribution, n_objs)
    expected = measure_alternative_impl(distribution)
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "disorder_measure, disorder_measure_alternative_impl",
    [
        (gini_impurity, gini_impurity_alternative_impl),
        (entropy, entropy_alternative_impl),
        (conflicts_count, conflicts_count_alternative_impl),
    ],
)
@pytest.mark.parametrize(
    "distribution",
    [
        [[0]],
        [[0]] * 10,
        [[4]],
        [[0, 0]],
        [[0, 0]] * 10,
        [[1, 1]],
        [[1, 1, 1]],
        [[4, 3, 3]],
        [[3, 1]],
        [[2, 0], [1, 1]],
        [[1, 1, 0, 0], [0, 0, 1, 1]],
        [[1, 1, 1, 1], [0, 0, 1, 1]],
        [[0, 0], [1, 2]],
        [[0, 2], [1, 1]],
        [[0, 1], [5, 1], [3, 5]],
        [[1, 1]] * 10,
        [[9999, 1]],
        [[9999, 1], [1, 1]],
        [[9999, 1], [1, 9999]],
        [[9999, 1], [0, 0]],
        [[9999, 1]] * 10,
    ],
)
def test_disorder_maesure(
    disorder_measure,
    disorder_measure_alternative_impl,
    distribution,
):
    run_compare_measure(
        distribution, disorder_measure, disorder_measure_alternative_impl
    )
