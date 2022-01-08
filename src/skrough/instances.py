import numpy as np


def draw_objects(group_index, dec_values, permutation=None):
    """
    Draw objects having uniform decision values within their groups
    """
    if permutation is None:
        permutation = np.random.permutation(len(group_index))
    _, idx = np.unique(group_index[permutation], return_index=True)
    idx = permutation[idx]
    group_dec_values = dict(zip(group_index[idx], dec_values[idx]))
    result = [
        i
        for i in range(len(group_index))
        if dec_values[i] == group_dec_values[group_index[i]]
    ]
    return result


# TODO: verify if it is better
# def draw_objects_new(group_index, dec_values, permutation=None):
#     """
#     Draw objects having uniform decision values within their groups
#     """
#     tab = np.concatenate(
#         [group_index[:, np.newaxis], dec_values[:, np.newaxis]], axis=1
#     )
#     if permutation is None:
#         permutation = np.random.permutation(tab.shape[0])
#     _, idx = np.unique(tab[permutation], return_index=True, axis=0)
#     idx = permutation[idx]
#     np.random.shuffle(idx)
#     group_dec_values = dict(tab[idx])
#     result = [
#         i
#         for i in range(len(group_index))
#         if dec_values[i] == group_dec_values[group_index[i]]
#     ]
#     return result
