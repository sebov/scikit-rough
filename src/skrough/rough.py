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
    attrs: rght.AttrsLike,
):
    group_index = GroupIndex.create_from_data(x, x_counts, attrs)
    dec_distribution = group_index.get_distribution(y, y_count)
    homogeneity = get_homogeneity(dec_distribution)
    # compute positions in `homogeneity` (here positions correspond to group ids) that
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
    attrs: rght.AttrsLike,
):
    if len(x) == 0:
        return 1
    pos = get_positive_region(x, x_counts, y, y_count, attrs)
    return len(pos) / len(x)


@numba.njit
def get_lower_upper_group_ids(distribution: np.ndarray):
    lower = []
    upper = []
    ngroup, ndec = distribution.shape
    for i in numba.prange(ngroup):
        if distribution[i, 1] > 0:
            upper.append(i)
            if distribution[i, 0] == 0:
                lower.append(i)
    return np.asarray(lower), np.asarray(upper)


def get_approximations(
    x: np.ndarray,
    x_counts: np.ndarray,
    objs: rght.ObjsLike,
    attrs: rght.AttrsLike,
):
    group_index = GroupIndex.create_from_data(x, x_counts, attrs)
    # treat membership as a decision attribute for this computation
    # imposed interpretation: 0 - not in objs, 1 - in obj
    membership = np.isin(np.arange(len(x)), objs).astype(int)
    membership_count = 2
    dec_distribution = group_index.get_distribution(membership, membership_count)
    lower_group_ids, upper_group_ids = get_lower_upper_group_ids(dec_distribution)
    lower = get_positions_where_values_in(group_index.index, lower_group_ids)
    upper = get_positions_where_values_in(group_index.index, upper_group_ids)
    return lower, upper
