from typing import Sequence, Union

import numpy as np

import skrough.typing as rght
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import GrowShrinkState, StateConfig


def grow_shrink(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    config: StateConfig,
    init_hooks: Union[rght.GSInitStateHook, Sequence[rght.GSInitStateHook]],
    stop_hook: rght.GSCheckStopHook,
    prepare_result_hook: rght.GSPrepareResultHook,
    seed: rght.Seed = None,
):
    rng = np.random.default_rng(seed)
    state = GrowShrinkState(
        group_index=GroupIndex.create_one_group(len(x)),
        rng=rng,
        config=config,
    )

    if not isinstance(init_hooks, Sequence):
        init_hooks = [init_hooks]

    for init_hook in init_hooks:
        init_hook(x, x_counts, y, y_count, state)

    while True:

        if stop_hook(x, x_counts, y, y_count, state):
            break

        candidate_attrs: np.ndarray = np.delete(
            np.arange(x.shape[1]),
            state.result_attrs,
        )

        attr = candidate_attrs[0]
        state.group_index = state.group_index.split(x[:, attr], x_counts[attr])
        state.result_attrs.append(attr)

    result = prepare_result_hook(x, x_counts, y, y_count, state)
    return result


#     # check stop condition

#     # get candidates

#     # check if candidates can be add

#     # reduction phase

#     # prepare result

#     # compute total chaos score

#     result = prepare_result(x, x_counts, y, y_count, state, config)
#     return result
