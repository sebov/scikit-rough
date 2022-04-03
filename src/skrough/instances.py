from typing import List, Optional

import numba
import numpy as np

import skrough.typing as rght
from skrough.permutations import get_permutation
from skrough.structs import GroupIndex
from skrough.unique import get_uniques_index
from skrough.weights import prepare_weights


@numba.njit
def get_pos_where_values_in(values, reference):
    """Get positions for which values are in the reference collection.

    Args:
        values: A collection of values for which to check if its elements are in the
            reference collection.
        reference: A collection of reference values that the values are checked against.

    Returns:
        A collection of indices for which a value on the given position is in
        the reference collection.
    """
    reference = set(reference)
    return [i for i in range(len(values)) if values[i] in reference]


def choose_objects(
    group_index: GroupIndex,
    y: np.ndarray,
    y_count: int,
    objs: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    seed: rght.Seed = None,
) -> List[int]:
    """
    Choose objects having uniform decision values within their groups.
    """

    # TODO: add arguments validation
    # 1) objs is not None => weights is None
    # 2) objs is None and weights is not None => len(group_index.index) == len(weights)

    if len(group_index.index) == 0:
        return []

    if objs is None:
        n = len(group_index.index)
        proba = prepare_weights(weights, n, expand_none=False)
        selector = get_permutation(0, n, proba, seed=seed)
    else:
        selector = np.asarray(objs)

    idx = get_uniques_index(group_index.index[selector])

    idx = selector[idx]
    group_index_dec = group_index.split(
        y,
        y_count,
        compress=False,
    )
    chosen = group_index_dec.index[idx]
    return get_pos_where_values_in(values=group_index_dec.index, reference=chosen)
