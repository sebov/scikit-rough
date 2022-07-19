import numpy as np
import pytest

from skrough.chaos_measures import conflicts_number, entropy, gini_impurity
from skrough.checks import (
    check_if_approx_reduct,
    check_if_consistent_table,
    check_if_functional_dependency,
    check_if_reduct,
)
from skrough.dataprep import prepare_factorized_array, prepare_factorized_vector


@pytest.mark.parametrize(
    "objs, attrs, expected_result",
    [
        (range(14), range(4), True),
        (None, None, True),
        ([], [], True),
        ([], None, True),
        ([], [0, 1], True),
        ([0], [0], True),
        ([1, 2], None, True),
        ([1, 2], [1, 2], False),
        ([0, 1, 5, 7, 13], [], True),
        ([2, 3, 4, 6, 8, 9, 10, 11, 12], [], True),
        (None, [0], False),
        (None, [0, 1], False),
        (None, [0, 1, 2], False),
        (None, [0, 1, 3], True),
        (None, [0, 2, 3], True),
        (None, [1, 2, 3], False),
    ],
)
def test_check_if_functional_dependency(
    objs, attrs, expected_result, golf_dataset_prep
):
    x, _, y, _ = golf_dataset_prep
    assert check_if_functional_dependency(x, y, objs, attrs) == expected_result
    if objs is not None:
        assert (
            check_if_functional_dependency(x, y, np.asarray(objs), attrs)
            == expected_result
        )
    if attrs is not None:
        assert (
            check_if_functional_dependency(x, y, objs, np.asarray(attrs))
            == expected_result
        )
    if objs is not None and attrs is not None:
        assert (
            check_if_functional_dependency(x, y, np.asarray(objs), np.asarray(attrs))
            == expected_result
        )


@pytest.mark.parametrize(
    "x, y, expected_result",
    [
        (
            [[0, 1], [1, 1]],
            [0, 1],
            True,
        ),
        (
            [[0, 0], [0, 0]],
            [0, 1],
            False,
        ),
        (
            [[], []],
            [0, 0],
            True,
        ),
        (
            [[], []],
            [0, 1],
            False,
        ),
    ],
)
def test_check_if_consistent_table(x, y, expected_result):
    x = np.asarray(x)
    y = np.asarray(y)
    assert check_if_consistent_table(x, y) == expected_result


@pytest.mark.parametrize(
    "x, y, attrs, consistent_table_check, expected_result",
    [
        (
            [[0, 1], [1, 1]],
            [0, 1],
            [0],
            False,
            True,
        ),
        (
            [[0, 1], [1, 1]],
            [0, 1],
            [0],
            True,
            True,
        ),
        (
            [[0, 1], [1, 1]],
            [0, 0],
            [0],
            False,
            False,
        ),
        (
            [[0, 1], [1, 1]],
            [0, 0],
            [0],
            True,
            False,
        ),
        (
            [[0, 1], [0, 1], [1, 0]],
            [0, 1, 1],
            [0],
            False,
            True,
        ),
        (
            [[0, 1], [0, 1], [1, 0]],
            [0, 1, 1],
            [0],
            True,
            False,
        ),
    ],
)
def test_check_if_reduct(x, y, attrs, consistent_table_check, expected_result):
    x = np.asarray(x)
    y = np.asarray(y)
    assert check_if_reduct(x, y, attrs, consistent_table_check) == expected_result


@pytest.mark.parametrize(
    "attrs, expected_result",
    [
        ([], False),
        ([0], False),
        ([1], False),
        ([2], False),
        ([3], False),
        ([0, 1], False),
        ([0, 2], False),
        ([0, 3], False),
        ([1, 2], False),
        ([1, 3], False),
        ([2, 3], False),
        ([0, 1, 2], False),
        ([0, 1, 3], True),
        ([0, 2, 3], True),
        ([1, 2, 3], False),
        ([0, 1, 2, 3], False),
    ],
)
def test_check_if_reduct_golf(attrs, expected_result, golf_dataset_prep):
    x, _, y, _ = golf_dataset_prep
    assert check_if_reduct(x, y, attrs) == expected_result


@pytest.mark.parametrize(
    "attrs",
    [
        [0, 0],
        [0, 1, 2, 3, 0],
        [0, 1, 2, 3, 3],
    ],
)
def test_check_if_reduct_golf_duplicated_attrs(attrs, golf_dataset_prep):
    x, _, y, _ = golf_dataset_prep
    with pytest.raises(ValueError, match="duplicated attrs"):
        check_if_reduct(x, y, attrs)


@pytest.mark.parametrize(
    "chaos_fun",
    [
        conflicts_number,
        entropy,
        gini_impurity,
    ],
)
@pytest.mark.parametrize(
    "epsilon, expected_is_superreduct",
    [
        (0, False),
        (0.1, False),
        (0.2, False),
        (0.8, True),
        (0.9, True),
        (1, True),
    ],
)
def test_check_if_approx_superreduct(epsilon, expected_is_superreduct, chaos_fun):
    x, x_counts = prepare_factorized_array(
        np.asarray(
            [
                [1, 0],
                [1, 1],
                [0, 2],
                [0, 3],
                [1, 4],
                [1, 5],
                [1, 6],
                [1, 7],
            ]
        )
    )
    y, y_count = prepare_factorized_vector(np.asarray([0, 0, 0, 0, 1, 1, 1, 1]))
    result = check_if_approx_reduct(
        x,
        x_counts,
        y,
        y_count,
        attrs=[0],
        chaos_fun=chaos_fun,
        epsilon=epsilon,
        check_attrs_reduction=False,
    )
    assert result == expected_is_superreduct


@pytest.mark.parametrize(
    "chaos_fun",
    [
        conflicts_number,
        entropy,
        gini_impurity,
    ],
)
@pytest.mark.parametrize(
    "epsilon",
    [0, 0.1, 0.2, 0.8, 0.9, 1],
)
def test_check_if_approx_reduct(epsilon, chaos_fun):
    x, x_counts = prepare_factorized_array(
        np.asarray(
            [
                [1, 1, 0],
                [0, 1, 1],
                [0, 0, 2],
                [0, 0, 3],
                [1, 1, 4],
                [1, 1, 5],
                [1, 1, 6],
                [1, 1, 7],
            ]
        )
    )
    y, y_count = prepare_factorized_vector(np.asarray([0, 0, 0, 0, 1, 1, 1, 1]))
    result = check_if_approx_reduct(
        x,
        x_counts,
        y,
        y_count,
        attrs=[0, 1],
        chaos_fun=chaos_fun,
        epsilon=epsilon,
        check_attrs_reduction=True,
    )
    assert result is False


@pytest.mark.parametrize(
    "chaos_fun, attrs, epsilon, check_attrs_reduction, expected",
    [
        (conflicts_number, [0, 1], 0, False, False),
        (conflicts_number, [0, 1], 0, True, False),
        (conflicts_number, [0, 1], 0.25, False, True),
        (conflicts_number, [0, 1], 0.25, True, True),
        (conflicts_number, [0, 1], 0.5, False, True),
        (conflicts_number, [0, 1], 0.5, True, False),
        (conflicts_number, [0, 1], 0.75, False, True),
        (conflicts_number, [0, 1], 0.75, True, False),
        (conflicts_number, [0, 1], 1, False, True),
        (conflicts_number, [0, 1], 1, True, False),
        #
        (conflicts_number, [0], 0, False, False),
        (conflicts_number, [0], 0, True, False),
        (conflicts_number, [0], 0.25, False, False),
        (conflicts_number, [0], 0.25, True, False),
        (conflicts_number, [0], 0.5, False, True),
        (conflicts_number, [0], 0.5, True, True),
        (conflicts_number, [0], 0.75, False, True),
        (conflicts_number, [0], 0.75, True, True),
        (conflicts_number, [0], 1, False, True),
        (conflicts_number, [0], 1, True, False),
        #
        (conflicts_number, [1], 0, False, False),
        (conflicts_number, [1], 0, True, False),
        (conflicts_number, [1], 0.25, False, False),
        (conflicts_number, [1], 0.25, True, False),
        (conflicts_number, [1], 0.5, False, True),
        (conflicts_number, [1], 0.5, True, True),
        (conflicts_number, [1], 0.75, False, True),
        (conflicts_number, [1], 0.75, True, True),
        (conflicts_number, [1], 1, False, True),
        (conflicts_number, [1], 1, True, False),
        #
        (entropy, [0, 1], 0, False, False),
        (entropy, [0, 1], 0, True, False),
        (entropy, [0, 1], 0.25, False, False),
        (entropy, [0, 1], 0.25, True, False),
        (entropy, [0, 1], 0.5, False, True),
        (entropy, [0, 1], 0.5, True, True),
        (entropy, [0, 1], 0.75, False, True),
        (entropy, [0, 1], 0.75, True, False),
        (entropy, [0, 1], 1, False, True),
        (entropy, [0, 1], 1, True, False),
        #
        (entropy, [0], 0, False, False),
        (entropy, [0], 0, True, False),
        (entropy, [0], 0.25, False, False),
        (entropy, [0], 0.25, True, False),
        (entropy, [0], 0.5, False, False),
        (entropy, [0], 0.5, True, False),
        (entropy, [0], 0.75, False, True),
        (entropy, [0], 0.75, True, True),
        (entropy, [0], 1, False, True),
        (entropy, [0], 1, True, False),
        #
        (entropy, [1], 0, False, False),
        (entropy, [1], 0, True, False),
        (entropy, [1], 0.25, False, False),
        (entropy, [1], 0.25, True, False),
        (entropy, [1], 0.5, False, False),
        (entropy, [1], 0.5, True, False),
        (entropy, [1], 0.75, False, False),
        (entropy, [1], 0.75, True, False),
        (entropy, [1], 1, False, True),
        (entropy, [1], 1, True, False),
        #
        (gini_impurity, [0, 1], 0, False, False),
        (gini_impurity, [0, 1], 0, True, False),
        (gini_impurity, [0, 1], 0.25, False, False),
        (gini_impurity, [0, 1], 0.25, True, False),
        (gini_impurity, [0, 1], 0.5, False, True),
        (gini_impurity, [0, 1], 0.5, True, True),
        (gini_impurity, [0, 1], 0.75, False, True),
        (gini_impurity, [0, 1], 0.75, True, False),
        (gini_impurity, [0, 1], 1, False, True),
        (gini_impurity, [0, 1], 1, True, False),
        #
        (gini_impurity, [0], 0, False, False),
        (gini_impurity, [0], 0, True, False),
        (gini_impurity, [0], 0.25, False, False),
        (gini_impurity, [0], 0.25, True, False),
        (gini_impurity, [0], 0.5, False, False),
        (gini_impurity, [0], 0.5, True, False),
        (gini_impurity, [0], 0.75, False, True),
        (gini_impurity, [0], 0.75, True, True),
        (gini_impurity, [0], 1, False, True),
        (gini_impurity, [0], 1, True, False),
        #
        (gini_impurity, [1], 0, False, False),
        (gini_impurity, [1], 0, True, False),
        (gini_impurity, [1], 0.25, False, False),
        (gini_impurity, [1], 0.25, True, False),
        (gini_impurity, [1], 0.5, False, False),
        (gini_impurity, [1], 0.5, True, False),
        (gini_impurity, [1], 0.75, False, False),
        (gini_impurity, [1], 0.75, True, False),
        (gini_impurity, [1], 1, False, True),
        (gini_impurity, [1], 1, True, False),
    ],
)
def test_check_if_approx_reduct_2(
    chaos_fun,
    attrs,
    epsilon,
    check_attrs_reduction,
    expected,
):
    x, x_counts = prepare_factorized_array(
        np.asarray(
            [
                [1, 1, 0],
                [0, 1, 1],
                [0, 0, 2],
                [0, 0, 3],
            ]
        )
    )
    y, y_count = prepare_factorized_vector(np.asarray([1, 0, 1, 0]))
    result = check_if_approx_reduct(
        x,
        x_counts,
        y,
        y_count,
        attrs=attrs,
        chaos_fun=chaos_fun,
        epsilon=epsilon,
        check_attrs_reduction=check_attrs_reduction,
    )
    assert result == expected
