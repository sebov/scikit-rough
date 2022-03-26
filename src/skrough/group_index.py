import numpy as np
import pandas.core.sorting

from skrough.containers import GroupIndex


def split_groups(
    group_index: GroupIndex,
    values: np.ndarray,
    values_count: int,
    compress_group_index: bool = True,
) -> GroupIndex:
    """
    Split groups of objects into finer groups according to values on
    a single splitting attribute
    """
    result = GroupIndex.create_empty()
    result.index = group_index.index * values_count + values
    result.count = group_index.count * values_count
    if compress_group_index:
        result = result.compress_group_index()
    return result


# TODO: check if it its actually slower
# def batch_split_into_groups_slower(xx, xx_count_distinct, attrs):
#     """
#     Split objects into groups according to values on given attributes
#     """
#     group_index = np.zeros(len(xx), dtype=np.int_)
#     n_groups = 1
#     for attr in attrs:
#         group_index, n_groups = split_groups(
#             group_index,
#             n_groups,
#             xx[:, attr],
#             xx_count_distinct[attr],
#             compress_group_index=True,
#         )
#     return group_index, n_groups


def batch_split_into_groups(
    x: np.ndarray,
    x_counts: np.ndarray,
    attrs: list[int],
) -> GroupIndex:
    """
    Split objects into groups according to values on given attributes
    """
    attrs = list(attrs)
    if not attrs:
        result = GroupIndex.create_one_group(size=len(x))
    else:
        result = GroupIndex.create_empty()
        result.index = pandas.core.sorting.get_group_index(
            labels=x[:, attrs].T,
            shape=x_counts[attrs],
            sort=False,
            xnull=False,
        )
        result = result.compress_group_index()
    return result
