import numpy as np


def draw_objects(group_index_index, y, permutation=None, seed=None):
    """
    Draw objects having uniform decision values within their groups
    """
    if permutation is None:
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(group_index_index))
    _, idx = np.unique(group_index_index[permutation], return_index=True)
    idx = permutation[idx]
    group_dec_values = dict(zip(group_index_index[idx], y[idx]))
    result = [
        i
        for i in range(len(group_index_index))
        if y[i] == group_dec_values[group_index_index[i]]
    ]
    return result
