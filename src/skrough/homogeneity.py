"""Functions related to homogeneity/heterogeneity of decision tables."""

from typing import Tuple

import numba
import numpy as np
import numpy.typing as npt

import skrough.typing as rght
from skrough.structs.group_index import GroupIndex
from skrough.unique import get_uniques_and_compacted


@numba.njit
def get_homogeneity(
    distribution: npt.NDArray[np.int64],
) -> npt.NDArray[np.int8]:
    """Compute distribution homogeneity.

    Compute homogeneity for a given input distribution. The function is mainly used for
    computation of homogeneity of decision attributes. The distribution format is
    defined as a 2D array where:

    - rows correspond to separate contexts, e.g., groups of objects or equivalence
      classes,
    - values in columns for a particular row represent discrete distribution, i.e.,
      the number of occurrences of each possible decision attribute distinct value.

    The result is a sequence of integer values (``0`` or ``1``), where each corresponds
    to a group/context (row) in the ``distribution`` input. A value of ``1`` means that
    there is at most one non-zero value in a given row (meaning that a row is
    homogenous), ``0`` otherwise (non-homogenous).

    Args:
        distribution: A 2D array representing a distribution.

    Raises:
        ValueError: If ``distribution`` is not a two-dimensional array.

    Returns:
        An array consisting of integer values ``0`` or ``1`` indicating that a
        corresponding row in the ``distribution`` input argument is either
        non-homogenous (for ``0``) or homogenous (for ``1``).

    Examples:
        >>> get_homogeneity(
        ...     np.asarray(
        ...         [
        ...             [0, 0],
        ...             [1, 1],
        ...             [0, 3],
        ...             [5, 0],
        ...         ]
        ...     )
        ... )
        array([1, 0, 1, 1])
    """
    if distribution.ndim != 2:
        raise ValueError("input `distribution` should be 2D")
    ngroup, ndec = distribution.shape
    result: npt.NDArray[np.int8] = np.ones(ngroup, dtype=np.int8)
    for i in numba.prange(ngroup):  # pylint: disable=not-an-iterable
        non_zero_so_far = False
        for j in range(ndec):
            if distribution[i, j] > 0:
                if non_zero_so_far:
                    result[i] = 0
                    break
                non_zero_so_far = True
    return result


HETEROGENEITY_MAX_COLS = 63


@numba.njit
def get_heterogeneity(
    distribution: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Compute distribution heterogeneity.

    Compute heterogeneity for a given input distribution. The function is mainly used
    for computation of heterogeneity of decision attributes. The distribution format is
    defined as a 2D array where:

    - rows correspond to separate contexts, e.g., groups of objects or equivalence
      classes,
    - values in columns for a particular row represent discrete distribution, i.e.,
      the number of occurrences of each possible decision attribute distinct value.

    The result is a sequence of integer values (``0`` or :code:`>=1`), where each
    corresponds to a group/context (row) in the ``distribution`` input. A value of ``0``
    means that there is at most one non-zero value in a given row (meaning that a row is
    non-heterogenous/homogenous). Values :code:`>=1` represent heterogenous rows, where
    different positive values show different kinds of heterogeneity. E.g., the function
    distinguishes a row where there are non zero values on positions ``0`` and ``1``
    from a row where there are non zero values on positions ``1`` and ``2``. The actual
    return value :code:`>=1` that corresponds to a given row is created as a binary
    represented number with bits set for places where discrete distribution counts are
    greater than ``0``.

    Args:
        distribution: A 2D array representing a distribution.

    Raises:
        ValueError: If ``distribution`` is not a two-dimensional array.
        ValueError: If the number of columns in the ``distribution`` input argument is
            greater than ``63``.

    Returns:
        An array consisting of integer values ``0`` or :code:`>=1` indicating that a
        corresponding row in the ``distribution`` input argument is either
        non-heterogenous/homogenous (for ``0``) or heterogenous (for :code:`>=1`).

    Examples:
        >>> get_heterogeneity(
        ...     np.asarray(
        ...         [
        ...             [0, 0, 0],
        ...             [1, 0, 0],
        ...             [0, 1, 0],
        ...             [0, 0, 1],
        ...             [1, 1, 0],
        ...             [1, 9, 0],
        ...             [9, 1, 0],
        ...             [1, 0, 1],
        ...             [1, 0, 9],
        ...             [9, 0, 1],
        ...             [0, 1, 1],
        ...             [0, 9, 1],
        ...             [0, 1, 9],
        ...             [1, 1, 1],
        ...             [1, 8, 9],
        ...             [8, 9, 1],
        ...         ]
        ...     )
        ... )
        array([0, 0, 0, 0, 6, 6, 6, 5, 5, 5, 3, 3, 3, 7, 7, 7])
    """
    if distribution.ndim != 2:
        raise ValueError("input `distribution` should be 2D")
    if distribution.shape[1] > HETEROGENEITY_MAX_COLS:
        raise ValueError("number of columns in `distribution` is too large")
    ngroup, ndec = distribution.shape
    result: npt.NDArray[np.int64] = np.zeros(ngroup, dtype=np.int64)
    for i in numba.prange(ngroup):  # pylint: disable=not-an-iterable
        non_zero_values = 0
        heterogeneity_value = np.int64(0)
        for j in range(ndec):
            heterogeneity_value = 2 * heterogeneity_value
            if distribution[i, j] > 0:
                non_zero_values += 1
                heterogeneity_value += 1
        if non_zero_values > 1:
            result[i] = heterogeneity_value
    return result


@numba.njit
def _replace_decisions_in_groups(
    group_ids: np.ndarray,
    y: np.ndarray,
    y_count: int,
    group_decisions: np.ndarray,
) -> np.ndarray:
    result = np.empty_like(y)
    for i in numba.prange(len(y)):  # pylint: disable=not-an-iterable
        if group_decisions[group_ids[i]] == 0:
            # ``0`` is reserved for non-heterogenous groups, so we preserve the original
            # decision
            result[i] = y[i]
        else:
            # values > ``0`` represents heterogenous groups, so we set new decisions for
            # objects belonging to those groups based on ``heterogeneity_values``; but
            # the new decision values need to be numbered accordingly, i.e., the values
            # need to be shifted behind the original range of decisions
            # ``0..(y_count-1)``
            result[i] = y_count - 1 + group_decisions[group_ids[i]]
    return result


def replace_heterogeneous_decisions(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    attrs: rght.LocationsLike,
    distinguish_generalized_decisions: bool = False,
) -> Tuple[np.ndarray, int]:
    """Return consistent decision values.

    Prepare new decision values in a way that makes data consistent (in the meaning of a
    consistent decision table). The groups, i.e., equivalence classes in the context of
    the indiscernibility relation, are induced from the given dataset ``x`` and a subset
    of attributes ``attrs``. Original decisions ``y`` are then processed to prepare new
    consistent decision values. It is done by preserving decision values for ...

    Args:
        x: _description_
        x_counts: _description_
        y: _description_
        y_count: _description_
        attrs: _description_
        distinguish_generalized_decisions: _description_. Defaults to False.

    Returns:
        _description_
    """
    if len(x) == 0:
        return y, y_count

    group_index = GroupIndex.from_data(x, x_counts, attrs)
    dec_distribution = group_index.get_distribution(y, y_count)
    if distinguish_generalized_decisions:
        heterogeneity = get_heterogeneity(dec_distribution)
    else:
        heterogeneity = 1 - get_homogeneity(dec_distribution)

    # values ``0`` (if present) mean non-heterogenous groups, i.e., homogenous groups
    # values > ``0`` (if present) mean heterogenous groups
    # np.unique returns sorted unique elements
    heterogeneity_unique_values, heterogeneity_compacted = get_uniques_and_compacted(
        heterogeneity
    )
    # let's compute the number of heterogenous groups
    heterogenous_groups_count = len(heterogeneity_unique_values)
    if heterogeneity_unique_values[0] == 0:
        # if the value ``0`` is there, we need to adjust, i.e.,
        # decrease heterogenous_group_count by 1
        heterogenous_groups_count -= 1
    else:
        # otherwise, we need to adjust ``heterogeneity_compacted``, as ``0`` now
        # represents actual heterogenous group but we want to keep ``0`` reserved for
        # non-heterogenous ones
        heterogeneity_compacted += 1

    result = _replace_decisions_in_groups(
        group_ids=group_index.index,
        y=y,
        y_count=y_count,
        group_decisions=heterogeneity_compacted,
    )

    return result, (y_count + heterogenous_groups_count)
