import numpy as np
import pandas.core.sorting


def split_groups(
    group_index,
    n_groups,
    factorized_values,
    values_count_distinct,
    compress_group_index=True,
):
    """
    Split groups of objects into finer groups according to values on
    a single splitting attribute
    """
    group_index = group_index * values_count_distinct + factorized_values
    if compress_group_index:
        group_index, n_groups = pandas.core.sorting.compress_group_index(
            group_index, sort=False
        )
        n_groups = len(n_groups)
    else:
        n_groups = n_groups * values_count_distinct
    return group_index, n_groups


# TODO: check if it its actually slower
def batch_split_into_groups_slower(xx, xx_count_distinct, attrs):
    """
    Split objects into groups according to values on given attributes
    """
    group_index = np.zeros(len(xx), dtype=np.int_)
    n_groups = 1
    for attr in attrs:
        group_index, n_groups = split_groups(
            group_index,
            n_groups,
            xx[:, attr],
            xx_count_distinct[attr],
            compress_group_index=True,
        )
    return group_index, n_groups


def batch_split_into_groups(xx, xx_count_distinct, attrs):
    """
    Split objects into groups according to values on given attributes
    """
    attrs = list(attrs)
    if not attrs:
        group_index, n_groups = np.zeros(len(xx), dtype=np.int_), 1
    else:
        group_index = pandas.core.sorting.get_group_index(
            labels=xx[:, attrs].T,
            shape=xx_count_distinct[attrs],
            sort=False,
            xnull=False,
        )
        group_index, n_groups = pandas.core.sorting.compress_group_index(
            group_index=group_index, sort=False
        )
        n_groups = len(n_groups)
    return group_index, n_groups
