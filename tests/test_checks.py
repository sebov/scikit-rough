import numpy as np
import pytest

from skrough.checks import check_if_consistent_table, check_if_functional_dependency


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
def test_checks_if_functional_dependency(
    objs, attrs, expected_result, golf_dataset_prep
):
    x, _, y, _ = golf_dataset_prep
    assert check_if_functional_dependency(x, y, objs, attrs) == expected_result


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
def test_checks_if_consistent_table(x, y, expected_result):
    x = np.asarray(x)
    y = np.asarray(y)
    assert check_if_consistent_table(x, y) == expected_result


def test_checks_reduct(golf_dataset_prep):
    assert len(golf_dataset_prep) == 4


def test_checks_bireduct(golf_dataset_prep):
    assert len(golf_dataset_prep) == 4
