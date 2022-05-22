from typing import Sequence, Tuple

import numba
import numpy as np
import numpy.typing as npt

from skrough.distributions import get_dec_distribution
from skrough.structs.group_index import GroupIndex


@numba.njit
def get_homogeneity(
    distribution: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """
    Compute decision homogeneity within groups of objects
    """
    if len(distribution.shape) != 2:
        raise ValueError("get_homogeneity - input distribution should be 2d")
    ngroup, ndec = distribution.shape
    result: npt.NDArray[np.int64] = np.ones(ngroup, dtype=np.int64)
    for i in numba.prange(ngroup):
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
    """
    Compute decision homogeneity within groups of objects
    """
    if len(distribution.shape) != 2:
        raise ValueError("get_heterogeneity - input distribution should be 2d")
    if distribution.shape[1] > HETEROGENEITY_MAX_COLS:
        raise ValueError("get_heterogeneity - number of columns loo large")
    ngroup, ndec = distribution.shape
    result: npt.NDArray[np.int64] = np.zeros(ngroup, dtype=np.int64)
    for i in numba.prange(ngroup):
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
    for i in numba.prange(len(y)):
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
    attrs: Sequence[int],
    distinguish_generalized_decisions: bool = False,
) -> Tuple[np.ndarray, int]:
    if len(x) == 0:
        y, y_count

    group_index = GroupIndex.create_from_data(x, x_counts, attrs)
    dec_distribution = get_dec_distribution(group_index, y, y_count)
    if distinguish_generalized_decisions:
        heterogeneity = get_heterogeneity(dec_distribution)
    else:
        heterogeneity = 1 - get_homogeneity(dec_distribution)

    # values ``0`` (if present) mean non-heterogenous groups, i.e., homogenous groups
    # values > ``0`` (if present) mean heterogenous groups
    # np.unique returns sorted unique elements
    heterogeneity_unique_values, heterogeneity_compacted = np.unique(
        heterogeneity, return_inverse=True
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
