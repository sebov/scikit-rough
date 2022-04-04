import itertools
from typing import Sequence, Union

import numpy as np
import pandas as pd

import skrough.typing as rght
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import GrowShrinkState, StateConfig


class LoopBreak(Exception):
    ...


def grow_shrink(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    config: StateConfig,
    init_hooks: Union[
        rght.GSInitStateHook,
        Sequence[rght.GSInitStateHook],
    ],
    check_stop_hooks: Union[
        rght.GSCheckStopHook,
        Sequence[rght.GSCheckStopHook],
    ],
    candidate_attrs_hooks: Union[
        rght.GSCandidateAttrsHook,
        Sequence[rght.GSCandidateAttrsHook],
    ],
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

    if not isinstance(check_stop_hooks, Sequence):
        check_stop_hooks = [check_stop_hooks]

    if not isinstance(candidate_attrs_hooks, Sequence):
        candidate_attrs_hooks = [candidate_attrs_hooks]

    for init_hook in init_hooks:
        init_hook(x, x_counts, y, y_count, state)

    # grow phase
    try:

        while True:

            if any(
                check_stop_hook(x, x_counts, y, y_count, state)
                for check_stop_hook in check_stop_hooks
            ):
                raise LoopBreak()

            remaining_attrs: np.ndarray = np.delete(
                np.arange(x.shape[1]),
                state.result_attrs,
            )

            candidate_attrs = np.fromiter(
                itertools.chain.from_iterable(
                    candidate_attrs_hook(
                        x, x_counts, y, y_count, state, remaining_attrs
                    )
                    for candidate_attrs_hook in candidate_attrs_hooks
                ),
                dtype=np.int64,
            )
            # remove duplicates, preserve order of appearance
            candidate_attrs = pd.unique(candidate_attrs)

            attr = candidate_attrs[0]
            state.group_index = state.group_index.split(x[:, attr], x_counts[attr])
            state.result_attrs.append(attr)

    except LoopBreak:
        pass

    # shrink phase

    # ...

    # prepare result

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
