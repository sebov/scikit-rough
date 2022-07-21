import numpy as np

import skrough.typing as rght
from skrough.structs.group_index import GroupIndex


def split_groups_and_compute_chaos_score(
    group_index: GroupIndex,
    attr_values: np.ndarray,
    attr_count: int,
    y: np.ndarray,
    y_count: int,
    chaos_fun: rght.ChaosMeasure,
):
    tmp_group_index = group_index.split(attr_values, attr_count)
    return tmp_group_index.get_chaos_score(y, y_count, chaos_fun)
