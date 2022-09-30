"""Test reducts"""

import numpy as np
import pytest

from skrough.dataprep import prepare_factorized_array, prepare_factorized_vector
from skrough.homogeneity import (
    HETEROGENEITY_MAX_COLS,
    get_heterogeneity,
    get_homogeneity,
    replace_heterogeneous_decisions,
)
from tests.helpers import generate_data


@pytest.mark.parametrize(
    "distribution, expected",
    [
        ([[2, 0], [0, 0], [1, 1]], [1, 1, 0]),
        ([[1, 1], [3, 2], [2, 2]], [0, 0, 0]),
        (
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 2, 0],
                [0, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
                [0, 2, 2],
                [3, 0, 2],
                [6, 1, 0],
                [3, 1, 2],
            ],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ),
    ],
)
def test_get_homogeneity(distribution, expected):
    distribution = np.asarray(distribution)
    result = get_homogeneity(distribution)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "distribution, error_match",
    [
        (generate_data(size=(0,)), "input `distribution` should be 2D"),
        (generate_data(size=(1,)), "input `distribution` should be 2D"),
        (generate_data(size=(0, 0, 0)), "input `distribution` should be 2D"),
        (generate_data(size=(0, 1, 0)), "input `distribution` should be 2D"),
        (generate_data(size=(0, 0, 3)), "input `distribution` should be 2D"),
        (generate_data(size=(1, 2, 3)), "input `distribution` should be 2D"),
    ],
)
def test_get_homogeneity_wrong_args(distribution, error_match):
    with pytest.raises(ValueError, match=error_match):
        distribution = np.asarray(distribution)
        get_homogeneity(distribution)


@pytest.mark.parametrize(
    "distribution, expected",
    [
        ([[2, 0], [0, 0], [1, 1]], [0, 0, 3]),
        ([[1, 1], [3, 2], [2, 2]], [3, 3, 3]),
        (
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 2, 0],
                [0, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
                [0, 2, 2],
                [3, 0, 2],
                [6, 1, 0],
                [3, 1, 2],
            ],
            [0, 0, 0, 0, 3, 5, 6, 7, 3, 5, 6, 7],
        ),
    ],
)
def test_get_heterogeneity(distribution, expected):
    distribution = np.asarray(distribution)
    result = get_heterogeneity(distribution)
    assert all([np.array_equal(result, expected)])


@pytest.mark.parametrize(
    "distribution, error_match",
    [
        (
            generate_data(size=(0,)),
            "input `distribution` should be 2D",
        ),
        (
            generate_data(size=(1,)),
            "input `distribution` should be 2D",
        ),
        (
            generate_data(size=(0, 0, 0)),
            "input `distribution` should be 2D",
        ),
        (
            generate_data(size=(0, 1, 0)),
            "input `distribution` should be 2D",
        ),
        (
            generate_data(size=(0, 0, 3)),
            "input `distribution` should be 2D",
        ),
        (
            generate_data(size=(1, 2, 3)),
            "input `distribution` should be 2D",
        ),
        (
            generate_data(size=(0, HETEROGENEITY_MAX_COLS + 1)),
            "number of columns in `distribution` is too large",
        ),
        (
            generate_data(size=(10, HETEROGENEITY_MAX_COLS + 1)),
            "number of columns in `distribution` is too large",
        ),
    ],
)
def test_get_heterogeneity_wrong_args(distribution, error_match):
    with pytest.raises(ValueError, match=error_match):
        distribution = np.asarray(distribution)
        get_heterogeneity(distribution)


def run_replace_heterogenous_decisions(data, dec, attrs, distinguish):
    x, x_counts = prepare_factorized_array(np.asarray(data))
    y, y_count = prepare_factorized_vector(dec)
    result = replace_heterogeneous_decisions(
        x,
        x_counts,
        y,
        y_count,
        attrs,
        distinguish_generalized_decisions=distinguish,
    )
    return result


replace_heterogenous_decisions_data = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 1],
    [0, 1, 1],
    [0, 1, 1],
    [0, 2, 1],
    [1, 2, 2],
    [1, 2, 0],
    [1, 2, 0],
]


@pytest.mark.parametrize(
    "data, dec, attrs, expected_y, expected_y_count",
    [
        (
            generate_data(size=(0, 0)),
            [],
            [],
            [],
            0,
        ),
        (
            generate_data(size=(0, 4)),
            [],
            [0, 1],
            [],
            0,
        ),
        (
            replace_heterogenous_decisions_data,
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            2,
        ),
        (
            replace_heterogenous_decisions_data,
            [0, 0, 0, 0, 0, 0, 0, 1, 2, 2],
            [0],
            [0, 0, 0, 0, 0, 0, 0, 3, 3, 3],
            4,
        ),
        (
            replace_heterogenous_decisions_data,
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
            [0],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            3,
        ),
        (
            replace_heterogenous_decisions_data,
            [0, 0, 1, 0, 0, 2, 0, 1, 2, 3],
            [1],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            5,
        ),
        (
            replace_heterogenous_decisions_data,
            [0, 0, 0, 1, 1, 2, 0, 0, 1, 2],
            [0, 1],
            [0, 0, 0, 3, 3, 3, 0, 3, 3, 3],
            4,
        ),
        (
            replace_heterogenous_decisions_data,
            [0, 1, 0, 1, 1, 1, 1, 0, 2, 3],
            [0, 1, 2],
            [4, 4, 0, 1, 1, 1, 1, 0, 4, 4],
            5,
        ),
    ],
)
def test_replace_heterogenous_decisions_no_distinguish(
    data,
    dec,
    attrs,
    expected_y,
    expected_y_count,
):
    new_y, new_y_count = run_replace_heterogenous_decisions(
        data,
        dec,
        attrs,
        distinguish=False,
    )
    assert new_y_count == expected_y_count
    assert np.array_equal(new_y, np.asarray(expected_y))


@pytest.mark.parametrize(
    "data, dec, attrs, expected_y, expected_y_count",
    [
        (
            generate_data(size=(0, 0)),
            [],
            [],
            [],
            0,
        ),
        (
            generate_data(size=(0, 4)),
            [],
            [0, 1],
            [],
            0,
        ),
        (
            replace_heterogenous_decisions_data,
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            2,
        ),
        (
            replace_heterogenous_decisions_data,
            [0, 0, 0, 0, 0, 0, 0, 1, 2, 2],
            [0],
            [0, 0, 0, 0, 0, 0, 0, 3, 3, 3],
            4,
        ),
        (
            replace_heterogenous_decisions_data,
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
            [0],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            3,
        ),
        (
            replace_heterogenous_decisions_data,
            [0, 0, 1, 0, 0, 2, 0, 1, 2, 3],
            [1],
            [5, 5, 5, 4, 4, 4, 6, 6, 6, 6],
            7,
        ),
        (
            replace_heterogenous_decisions_data,
            [0, 0, 0, 1, 1, 2, 2, 0, 1, 2],
            [0, 1],
            [0, 0, 0, 3, 3, 3, 2, 4, 4, 4],
            5,
        ),
        (
            replace_heterogenous_decisions_data,
            [0, 1, 2, 0, 1, 1, 3, 1, 1, 2],
            [0, 1, 2],
            [5, 5, 2, 5, 5, 5, 3, 1, 4, 4],
            6,
        ),
    ],
)
def test_replace_heterogenous_decisions_distinguish(
    data,
    dec,
    attrs,
    expected_y,
    expected_y_count,
):
    new_y, new_y_count = run_replace_heterogenous_decisions(
        data,
        dec,
        attrs,
        distinguish=True,
    )
    assert new_y_count == expected_y_count
    assert np.array_equal(new_y, np.asarray(expected_y))
