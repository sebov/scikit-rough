import numpy as np

import skrough.typing as rght
from skrough.chaos_score import get_chaos_score_for_group_index
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
    return get_chaos_score_for_group_index(tmp_group_index, y, y_count, chaos_fun)
