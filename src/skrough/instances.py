from typing import Optional

import numba
import numpy as np

import skrough as rgh
import skrough.typing as rght


@numba.njit
def get_pos_where_values_in(values, reference):
    """Get positions for which values are in the reference collection.

    Args:
        values: A collection of values for which to check if its elements are in the
            reference collection.
        reference: A collection of reference values that the values are checked against.

    Returns:
        A colection of indices for which a value on the given position is in
        the reference collection.
    """
    reference = set(reference)
    return [i for i in range(len(values)) if values[i] in reference]


def choose_objects(
    group_index: rgh.containers.GroupIndex,
    y: np.ndarray,
    y_count: int,
    objs: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    seed: rght.Seed = None,
):
    """
    Choose objects having uniform decision values within their groups.
    """

    # TODO: add arguments validation
    # 1) objs is not None => weights is None
    # 2) objs is None and weights is not None => len(group_index.index) == len(weights)

    if len(group_index.index) == 0:
        return []

    if objs is None:
        rng = np.random.default_rng(seed)
        n = len(group_index.index)
        proba = rgh.weights.prepare_weights(weights, n, expand_none=False)
        selector = rgh.permutations.draw_values(0, n, proba, seed=rng)
    else:
        selector = np.asarray(objs)

    _, idx = np.unique(group_index.index[selector], return_index=True)
    idx = selector[idx]
    group_index_dec = rgh.group_index.split_groups(
        group_index, y, y_count, compress_group_index=False
    )
    chosen = group_index_dec.index[idx]
    return get_pos_where_values_in(values=group_index_dec.index, reference=chosen)
