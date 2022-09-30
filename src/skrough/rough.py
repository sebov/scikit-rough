"""Rough sets related functions."""

from typing import List, Tuple

import numba
import numpy as np

import skrough.typing as rght
from skrough.homogeneity import get_homogeneity
from skrough.structs.group_index import GroupIndex
from skrough.utils import get_positions_where_values_in


def get_positive_region(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    attrs: rght.LocationsLike,
) -> List[int]:
    group_index = GroupIndex.from_data(x, x_counts, attrs)
    dec_distribution = group_index.get_distribution(y, y_count)
    homogeneity = get_homogeneity(dec_distribution)
    # compute positions in ``homogeneity`` (here positions correspond to group ids) that
    # are equal to True
    homogenous_groups = homogeneity.nonzero()[0]
    # return positions in group_index (they correspond to objects) for which values
    # belong to the set of homogenous groups
    return get_positions_where_values_in(group_index.index, homogenous_groups)


def get_gamma_value(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    attrs: rght.LocationsLike,
) -> float:
    if len(x) == 0:
        return 1
    pos = get_positive_region(x, x_counts, y, y_count, attrs)
    return len(pos) / len(x)


@numba.njit
def get_lower_upper_group_ids(
    membership_distr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if membership_distr.ndim != 2 or membership_distr.shape[1] != 2:
        raise ValueError(
            "Membership distribution should be a 2D array of just two columns"
        )
    lower = []
    upper = []
    ngroup = len(membership_distr)
    for i in numba.prange(ngroup):  # pylint: disable=not-an-iterable
        if membership_distr[i, 1] > 0:
            upper.append(i)
            if membership_distr[i, 0] == 0:
                lower.append(i)
    return np.asarray(lower), np.asarray(upper)


def get_approximations(
    x: np.ndarray,
    x_counts: np.ndarray,
    objs: rght.LocationsLike,
    attrs: rght.LocationsLike,
) -> Tuple[List[int], List[int]]:
    group_index = GroupIndex.from_data(x, x_counts, attrs)
    # treat membership as a decision attribute for this computation
    # imposed interpretation: 0 - not in objs, 1 - in obj
    membership = np.isin(np.arange(len(x)), objs).astype(int)
    membership_count = 2
    membership_distr = group_index.get_distribution(membership, membership_count)
    lower_group_ids, upper_group_ids = get_lower_upper_group_ids(membership_distr)
    lower = get_positions_where_values_in(group_index.index, lower_group_ids)
    upper = get_positions_where_values_in(group_index.index, upper_group_ids)
    return lower, upper
