from typing import List, Optional

import numpy as np

import skrough.typing as rght
from skrough.permutations import get_permutation
from skrough.structs.group_index import GroupIndex
from skrough.unique import get_uniques_index
from skrough.utils import get_positions_where_values_in
from skrough.weights import prepare_weights


def choose_objects(
    group_index: GroupIndex,
    y: np.ndarray,
    y_count: int,
    objs: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    return_representatives_only: bool = False,
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

    representatives_ids = selector[idx]

    if return_representatives_only:
        result = sorted(representatives_ids)
    else:
        group_index_dec = group_index.split(
            y,
            y_count,
            compress=False,
        )
        group_ids = group_index_dec.index[representatives_ids]
        result = get_positions_where_values_in(
            values=group_index_dec.index, reference=group_ids
        )

    return result
