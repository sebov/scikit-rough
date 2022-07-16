from typing import Literal, cast

import numpy as np
import pytest

from skrough.permutations import (
    get_attrs_permutation,
    get_objs_attrs_permutation,
    get_objs_permutation,
    get_permutation,
)


@pytest.mark.parametrize(
    "start, stop",
    [
        (0, 0),
        (1, 1),
        (1, 0),
        (2, 0),
        (5, 4),
        (-1, -1),
        (1, -10),
        (0, 1),
        (0, 10),
        (2, 7),
        (-1, 1),
        (-1, 5),
        (-10, 10),
    ],
)
def test_get_permutation(start, stop):
    result = get_permutation(start, stop)
    assert np.array_equal(np.unique(result), np.arange(start, stop))


@pytest.mark.parametrize(
    "n_objs, n_attrs",
    [
        (0, 0),
        (0, 1),
        (0, 10),
        (1, 0),
        (10, 0),
        (1, 1),
        (2, 2),
        (10, 10),
    ],
)
def test_get_objs_attrs_permutation(n_objs, n_attrs):
    for mode in ["mixed", "objs_before", "attrs_before"]:
        mode = cast(Literal["mixed", "objs_before", "attrs_before"], mode)
        result = get_objs_attrs_permutation(n_objs, n_attrs, strategy=mode)
        if mode == "mixed":
            assert np.array_equal(np.unique(result), np.arange(n_objs + n_attrs))
        elif mode == "objs_before":
            assert np.array_equal(np.unique(result[0:n_objs]), np.arange(n_objs))
            assert np.array_equal(
                np.unique(result[n_objs:]), np.arange(n_attrs) + n_objs
            )
        else:
            assert np.array_equal(
                np.unique(result[0:n_attrs]), np.arange(n_attrs) + n_objs
            )
            assert np.array_equal(np.unique(result[n_attrs:]), np.arange(n_objs))


@pytest.mark.parametrize(
    "n_objs, n_attrs, mode",
    [
        (-1, 0, "mixed"),
        (0, -1, "mixed"),
        (-1, -1, "mixed"),
        (-1, 0, "objs_before"),
        (0, -1, "objs_before"),
        (-1, -1, "objs_before"),
        (-1, 0, "attrs_before"),
        (0, -1, "attrs_before"),
        (-1, -1, "attrs_before"),
    ],
)
def test_get_objs_attrs_permutation_wrong_args(n_objs, n_attrs, mode):
    with pytest.raises(ValueError, match="cannot be less than zero"):
        get_objs_attrs_permutation(n_objs, n_attrs, strategy=mode)


@pytest.mark.parametrize(
    "mode",
    ["mixed000", "objects", "objs_after"],
)
def test_get_objs_attrs_permutation_wrong_mode(mode):
    with pytest.raises(ValueError, match="Unrecognized permutation strategy"):
        get_objs_attrs_permutation(0, 0, strategy=mode)


@pytest.mark.parametrize(
    "n_objs",
    [0, 1, 5],
)
def test_get_objs_permutation(n_objs):
    result = get_objs_permutation(n_objs=n_objs)
    assert np.array_equal(np.unique(result), np.arange(n_objs))


@pytest.mark.parametrize(
    "n_objs",
    [-1, -10],
)
def test_get_objs_permutation_wrong_args(n_objs):
    with pytest.raises(ValueError, match="`n_objs` cannot be less than zero"):
        get_objs_permutation(n_objs=n_objs)


@pytest.mark.parametrize(
    "n_attrs",
    [0, 1, 5],
)
def test_get_attrs_permutation(n_attrs):
    result = get_attrs_permutation(n_attrs=n_attrs)
    assert np.array_equal(np.unique(result), np.arange(n_attrs))


@pytest.mark.parametrize(
    "n_attrs",
    [-1, -10],
)
def test_get_attrs_permutation_wrong_args(n_attrs):
    with pytest.raises(ValueError, match="`n_attrs` cannot be less than zero"):
        get_attrs_permutation(n_attrs=n_attrs)
