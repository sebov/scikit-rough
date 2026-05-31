"""Test reducts"""

import numpy as np
import pytest

from skrough.dataprep import prepare_factorized_array, prepare_factorized_vector
from skrough.homogeneity import (
    HETEROGENEITY_MAX_COLS,
    encode_heterogeneity,
    encode_heterogeneity_alt,
    encode_homogeneity,
    heterogeneous_groups_decisions_replace,
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
def test_encode_homogeneity(distribution, expected):
    distribution = np.asarray(distribution)
    result = encode_homogeneity(distribution)
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
def test_encode_homogeneity_wrong_args(distribution, error_match):
    with pytest.raises(ValueError, match=error_match):
        distribution = np.asarray(distribution)
        encode_homogeneity(distribution)


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
def test_encode_heterogeneity(distribution, expected):
    distribution = np.asarray(distribution)
    result = encode_heterogeneity(distribution)
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
def test_encode_heterogeneity_wrong_args(distribution, error_match):
    with pytest.raises(ValueError, match=error_match):
        distribution = np.asarray(distribution)
        encode_heterogeneity(distribution)


def run_replace_heterogenous_decisions(data, dec, attrs, distinguish):
    x, x_counts = prepare_factorized_array(np.asarray(data))
    y, y_count = prepare_factorized_vector(dec)
    result = heterogeneous_groups_decisions_replace(
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


@pytest.mark.parametrize(
    "distribution, expected",
    [
        ([[2, 0], [0, 0], [1, 1]], [0, 0, 1]),
        ([[1, 1], [3, 2], [2, 2]], [1, 1, 1]),
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
            [0, 0, 0, 0, 1, 2, 3, 4, 1, 2, 3, 4],
        ),
    ],
)
def test_encode_heterogeneity_alt_indicator(distribution, expected):
    distribution = np.asarray(distribution)
    result = encode_heterogeneity_alt(distribution, use_indicator=True)
    heterogeneous_mask = result > 0
    expected_mask = np.asarray(expected) > 0
    assert np.array_equal(heterogeneous_mask, expected_mask)

    for val in np.unique(result):
        mask_result = result == val
        if val > 0:
            expected_vals = np.unique(np.asarray(expected)[mask_result])
            assert len(expected_vals) == 1, (
                f"Alt pattern {val} maps to multiple expected values: {expected_vals}"
            )


@pytest.mark.parametrize(
    "distribution",
    [
        [[2, 0], [0, 0], [1, 1]],
        [[1, 1], [3, 2], [2, 2]],
        [[2, 2], [3, 1], [1, 1]],
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
            ]
        ),
    ],
)
def test_encode_heterogeneity_alt_full_counts(distribution):
    distribution = np.asarray(distribution)
    result = encode_heterogeneity_alt(distribution, use_indicator=False)

    assert result.shape == (len(distribution),)
    assert np.all(result >= 0)

    non_zero_counts = np.sum(distribution > 0, axis=1)
    expected_heterogeneous = non_zero_counts > 1
    result_heterogeneous = result > 0
    assert np.array_equal(result_heterogeneous, expected_heterogeneous)

    unique_rows = np.unique(distribution, axis=0)
    for unique_row in unique_rows:
        mask = np.all(distribution == unique_row, axis=1)
        if np.sum(unique_row > 0) > 1:
            result_vals = np.unique(result[mask])
            assert len(result_vals) == 1, (
                f"Same distribution row {unique_row} maps to multiple values: {result_vals}"
            )

    for i in range(len(distribution)):
        for j in range(i + 1, len(distribution)):
            if not np.array_equal(distribution[i], distribution[j]):
                if np.sum(distribution[i] > 0) > 1 and np.sum(distribution[j] > 0) > 1:
                    assert result[i] != result[j], (
                        f"Different rows {distribution[i]} and {distribution[j]} got same value {result[i]}"
                    )


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
def test_encode_heterogeneity_alt_wrong_args(distribution, error_match):
    with pytest.raises(ValueError, match=error_match):
        distribution = np.asarray(distribution)
        encode_heterogeneity_alt(distribution)


def test_encode_heterogeneity_alt_100_columns():
    data = generate_data(size=(100, 100))
    result = encode_heterogeneity_alt(data)
    assert result.shape == (100,)
    assert np.all(result >= 0)

    heterogeneous_mask = result > 0
    expected_mask = np.sum(data > 0, axis=1) > 1
    assert np.array_equal(heterogeneous_mask, expected_mask)


def test_encode_heterogeneity_alt_100_columns_full_counts():
    data = generate_data(size=(50, 100))
    result = encode_heterogeneity_alt(data, use_indicator=False)
    assert result.shape == (50,)
    assert np.all(result >= 0)

    heterogeneous_mask = result > 0
    expected_mask = np.sum(data > 0, axis=1) > 1
    assert np.array_equal(heterogeneous_mask, expected_mask)


def test_encode_heterogeneity_alt_equivalence_with_fast():
    test_data = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 9, 0],
            [9, 1, 0],
            [1, 0, 1],
            [1, 0, 9],
            [9, 0, 1],
            [0, 1, 1],
            [0, 9, 1],
            [0, 1, 9],
            [1, 1, 1],
            [1, 8, 9],
            [8, 9, 1],
        ]
    )

    result_fast = encode_heterogeneity(test_data)
    result_alt = encode_heterogeneity_alt(test_data, use_indicator=True)

    fast_mask = result_fast > 0
    alt_mask = result_alt > 0
    assert np.array_equal(fast_mask, alt_mask)

    for val in np.unique(result_fast):
        mask_fast = result_fast == val
        if val > 0:
            alt_vals = np.unique(result_alt[mask_fast])
            assert len(alt_vals) == 1, (
                f"Fast pattern {val} maps to multiple alt values: {alt_vals}"
            )

    for val in np.unique(result_alt):
        mask_alt = result_alt == val
        if val > 0:
            fast_vals = np.unique(result_fast[mask_alt])
            assert len(fast_vals) == 1, (
                f"Alt pattern {val} maps to multiple fast values: {fast_vals}"
            )
