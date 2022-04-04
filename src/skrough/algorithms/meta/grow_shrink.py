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
    get_candidate_attrs_hooks: Union[
        rght.GSGetCandidateAttrsHook,
        Sequence[rght.GSGetCandidateAttrsHook],
    ],
    select_attrs_hook: rght.GSSelectAttrsHook,
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

    if not isinstance(get_candidate_attrs_hooks, Sequence):
        get_candidate_attrs_hooks = [get_candidate_attrs_hooks]

    def check_stop(check_stop_hooks: Sequence[rght.GSCheckStopHook]):
        # stop hooks
        if any(
            check_stop_hook(x, x_counts, y, y_count, state)
            for check_stop_hook in check_stop_hooks
        ):
            raise LoopBreak()

    # init hooks
    for init_hook in init_hooks:
        init_hook(x, x_counts, y, y_count, state)

    ################
    # grow phase
    ################
    try:

        state.values["empty_add_attrs_count"] = 0

        check_stop(check_stop_hooks)

        while True:

            remaining_attrs: np.ndarray = np.delete(
                np.arange(x.shape[1]),
                state.result_attrs,
            )

            if len(remaining_attrs) == 0:
                break

            # candidate attrs hooks
            candidate_attrs = np.fromiter(
                itertools.chain.from_iterable(
                    get_candidate_attrs_hook(
                        x, x_counts, y, y_count, state, remaining_attrs
                    )
                    for get_candidate_attrs_hook in get_candidate_attrs_hooks
                ),
                dtype=np.int64,
            )
            # remove duplicates, preserve order of appearance
            candidate_attrs = pd.unique(candidate_attrs)

            # select attrs hook
            selected_attrs = select_attrs_hook(
                x,
                x_counts,
                y,
                y_count,
                state,
                candidate_attrs,
            )

            if len(selected_attrs) == 0:
                check_stop(check_stop_hooks)
            else:
                for selected_attr in selected_attrs:
                    state.group_index = state.group_index.split(
                        x[:, selected_attr],
                        x_counts[selected_attr],
                    )
                    state.result_attrs.append(selected_attr)
                    check_stop(check_stop_hooks)

    except LoopBreak:
        pass

    ################
    # end grow phase
    ################

    ##################
    # shrink phase
    ##################

    # ...

    ##################
    # end shrink phase
    ##################

    # prepare result hook
    result = prepare_result_hook(x, x_counts, y, y_count, state)

    return result
