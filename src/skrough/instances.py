from typing import Optional

import numpy as np
import numpy.typing as npt

import skrough as rgh
import skrough.typing as rght
from skrough.containers import GroupIndex


def choose_objects_slower(
    group_index: GroupIndex,
    y: np.ndarray,
    permutation: Optional[npt.ArrayLike] = None,
    seed: rght.RandomState = None,
):
    """
    Choose objects having uniform decision values within their groups
    """
    if permutation is None:
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(group_index.index))
    else:
        permutation = np.array(permutation)
    _, idx = np.unique(group_index.index[permutation], return_index=True)
    idx = permutation[idx]
    group_dec_values = dict(zip(group_index.index[idx], y[idx]))
    result = [
        i
        for i in range(len(group_index.index))
        if y[i] == group_dec_values[group_index.index[i]]
    ]
    return result


def choose_objects(
    group_index,
    dec_values,
    dec_values_count,
    objs=None,
    weights=None,
    seed=None,
):
    """
    Choose objects having uniform decision values within their groups
    """

    # TODO: add arguments validation
    # 1) objs is not None => weights is None
    # 2) objs is None and weights is not None => len(group_index.index) == len(weights)

    if len(group_index.index) == 0:
        return []

    def normalize(weights):
        norm = np.linalg.norm(weights, ord=1)
        if norm > 0:
            weights = weights / norm
        return weights

    if weights is not None:
        weights = np.asarray(weights)
        weights = normalize(weights)
        if any(weights == 0):
            weights += np.finfo(dtype=np.float64).eps
            weights = normalize(weights)
        print(weights)

    if objs is None:
        rng = np.random.default_rng(seed)
        n = len(group_index.index)
        selector = rng.choice(n, size=n, replace=False, p=weights)
    else:
        selector = np.asarray(objs)

    _, idx = np.unique(group_index.index[selector], return_index=True)
    idx = selector[idx]
    tmp = rgh.group_index.split_groups(
        group_index, dec_values, dec_values_count, compress_group_index=False
    )
    tmp2 = tmp.index[idx]
    return np.arange(len(tmp.index))[np.isin(tmp.index, tmp2)].tolist()
