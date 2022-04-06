import itertools
import logging
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

import skrough.typing as rght
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import GrowShrinkState, StateConfig

logger = logging.getLogger(__name__)


class LoopBreak(Exception):
    ...


def grow_shrink(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    config: StateConfig,
    init_hooks: Optional[
        Union[
            rght.GSInitStateHook,
            Sequence[rght.GSInitStateHook],
        ]
    ],
    grow_stop_hooks: Union[
        rght.GSGrowStopHook,
        Sequence[rght.GSGrowStopHook],
    ],
    grow_candidate_attrs_hooks: Optional[
        Union[
            rght.GSGrowCandidateAttrsHook,
            Sequence[rght.GSGrowCandidateAttrsHook],
        ]
    ],
    grow_select_attrs_hook: rght.GSGrowSelectAttrsHook,
    grow_post_select_attrs_hooks: Optional[
        Union[
            rght.GSGrowPostSelectAttrsHook,
            Sequence[rght.GSGrowPostSelectAttrsHook],
        ]
    ],
    shrink_candidate_attrs_hook: Optional[rght.GSShrinkCandidateAttrsHook],
    shrink_accept_group_index_hooks: Optional[
        Union[
            rght.GSShrinkAcceptGroupIndexHook,
            Sequence[rght.GSShrinkAcceptGroupIndexHook],
        ]
    ],
    finalize_hooks: Optional[
        Union[
            rght.GSFinalizeStateHook,
            Sequence[rght.GSFinalizeStateHook],
        ]
    ],
    prepare_result_hook: rght.GSPrepareResultHook,
    seed: rght.Seed = None,
):
    logger.info("Start %s function", grow_shrink.__name__)

    logger.debug("Create state object")
    rng = np.random.default_rng(seed)
    state = GrowShrinkState(
        group_index=GroupIndex.create_one_group(len(x)),
        rng=rng,
        config=config,
    )

    logger.debug("Normalize init_hooks")
    if (init_hooks is not None) and (not isinstance(init_hooks, Sequence)):
        init_hooks = [init_hooks]

    logger.debug("Normalize grow_stop_hooks")
    if not isinstance(grow_stop_hooks, Sequence):
        grow_stop_hooks = [grow_stop_hooks]

    logger.debug("Normalize get_candidate_attrs_hooks")
    if (grow_candidate_attrs_hooks is not None) and (
        not isinstance(grow_candidate_attrs_hooks, Sequence)
    ):
        grow_candidate_attrs_hooks = [grow_candidate_attrs_hooks]

    logger.debug("Normalize post_select_attrs_hooks")
    if (grow_post_select_attrs_hooks is not None) and (
        not isinstance(grow_post_select_attrs_hooks, Sequence)
    ):
        grow_post_select_attrs_hooks = [grow_post_select_attrs_hooks]

    logger.debug("Normalize post_select_attrs_hooks")
    if (shrink_accept_group_index_hooks is not None) and (
        not isinstance(shrink_accept_group_index_hooks, Sequence)
    ):
        shrink_accept_group_index_hooks = [shrink_accept_group_index_hooks]

    logger.debug("Normalize finalize_hooks")
    if (finalize_hooks is not None) and (not isinstance(finalize_hooks, Sequence)):
        finalize_hooks = [finalize_hooks]

    def create_grow_stop_check(
        grow_stop_hooks: Sequence[rght.GSGrowStopHook],
    ):
        def _grow_stop_check(
            x: np.ndarray,
            x_counts: np.ndarray,
            y: np.ndarray,
            y_count: int,
            state: GrowShrinkState,
        ):
            if any(
                stop_hook(x, x_counts, y, y_count, state)
                for stop_hook in grow_stop_hooks
            ):
                raise LoopBreak()

        return _grow_stop_check

    grow_stop_check = create_grow_stop_check(grow_stop_hooks)

    def create_shrink_accept_group_index_check(
        shrink_accept_group_index_hooks: Optional[
            Sequence[rght.GSShrinkAcceptGroupIndexHook]
        ],
    ):
        def _shrink_accept_group_index_check(
            x: np.ndarray,
            x_counts: np.ndarray,
            y: np.ndarray,
            y_count: int,
            state: GrowShrinkState,
            group_index_to_check: GroupIndex,
        ):
            result = False
            if shrink_accept_group_index_hooks is not None:
                result = all(
                    accept_hook(x, x_counts, y, y_count, state, group_index_to_check)
                    for accept_hook in shrink_accept_group_index_hooks
                )
            return result

        return _shrink_accept_group_index_check

    shrink_accept_group_index_check = create_shrink_accept_group_index_check(
        shrink_accept_group_index_hooks
    )

    # init hooks
    logger.debug("Run init hooks")
    if init_hooks is not None:
        for init_hook in init_hooks:
            init_hook(x, x_counts, y, y_count, state)

    ################
    # grow phase
    ################
    logger.info("Start grow phase")
    try:

        logger.debug("Check stop conditions")
        grow_stop_check(x, x_counts, y, y_count, state)

        while True:

            logger.debug("Get remaining attributes")
            remaining_attrs: np.ndarray = np.delete(
                np.arange(x.shape[1]),
                state.result_attrs,
            )

            if len(remaining_attrs) == 0:
                logger.debug("No remaining attributes")
                break

            # candidate attrs hooks
            logger.debug("Obtain candidate attrs")
            if grow_candidate_attrs_hooks is None:
                logger.debug("Use all remaining attrs as candidate attrs")
                grow_candidate_attrs = remaining_attrs
            else:
                logger.debug("Obtain candidate attrs using candidate attrs hooks")
                grow_candidate_attrs = np.fromiter(
                    itertools.chain.from_iterable(
                        grow_candidate_attrs_hook(
                            x, x_counts, y, y_count, state, remaining_attrs
                        )
                        for grow_candidate_attrs_hook in grow_candidate_attrs_hooks
                    ),
                    dtype=np.int64,
                )
                # remove duplicates, preserve order of appearance
                logger.debug("Remove duplicates from candidate attrs")
                grow_candidate_attrs = pd.unique(grow_candidate_attrs)
            logger.debug("Grow candidate attrs count = %d", len(grow_candidate_attrs))

            # select attrs hook
            logger.debug("Select attrs using select attrs hooks")
            selected_attrs = grow_select_attrs_hook(
                x,
                x_counts,
                y,
                y_count,
                state,
                grow_candidate_attrs,
            )
            logger.info("Selected attrs = %s", selected_attrs)

            logger.debug("Run post select hooks")
            if grow_post_select_attrs_hooks is not None:
                for grow_post_select_attrs_hook in grow_post_select_attrs_hooks:
                    selected_attrs = grow_post_select_attrs_hook(
                        x,
                        x_counts,
                        y,
                        y_count,
                        state,
                        selected_attrs,
                    )
            logger.info("Selected attrs after post hooks = %s", selected_attrs)

            logger.debug("Process selected attrs")
            if len(selected_attrs) == 0:
                logger.debug("Empty selected attrs collection - check stop conditions")
                grow_stop_check(x, x_counts, y, y_count, state)
            else:
                logger.debug("Add selected attrs one by one")
                for selected_attr in selected_attrs:
                    logger.info("Add attr <%d>", selected_attr)
                    state.result_attrs.append(selected_attr)
                    state.group_index = state.group_index.split(
                        x[:, selected_attr],
                        x_counts[selected_attr],
                    )
                    logger.debug("Check stop conditions")
                    grow_stop_check(x, x_counts, y, y_count, state)

    except LoopBreak:
        pass
    logger.info("End grow phase")
    ################
    # end grow phase
    ################

    logger.info("Attrs after grow phase = %s", state.result_attrs)

    ##################
    # shrink phase
    ##################
    logger.info("Start shrink phase")

    if shrink_candidate_attrs_hook is None:
        shrink_candidate_attrs = np.asarray(list(reversed(state.result_attrs)))
    else:
        shrink_candidate_attrs = shrink_candidate_attrs_hook(
            x,
            x_counts,
            y,
            y_count,
            state,
        )

    logger.debug("Shrink candidate attrs count = %d", len(shrink_candidate_attrs))

    for shrink_candidate_attr in shrink_candidate_attrs:
        shrinked_attrs = state.result_attrs[:]
        shrinked_attrs.remove(shrink_candidate_attr)
        shrinked_group_index = GroupIndex.create_from_data(
            x,
            x_counts,
            shrinked_attrs,
        )
        if shrink_accept_group_index_check(
            x,
            x_counts,
            y,
            y_count,
            state,
            shrinked_group_index,
        ):
            logger.info("Removing attr <%d>", shrink_candidate_attr)
            state.result_attrs = shrinked_attrs
            state.group_index = shrinked_group_index

    logger.info("End shrink phase")
    ##################
    # end shrink phase
    ##################

    logger.debug("Run finalize hooks")
    if finalize_hooks is not None:
        for finalize_hook in finalize_hooks:
            finalize_hook(x, x_counts, y, y_count, state)

    # prepare result hook
    result = prepare_result_hook(x, x_counts, y, y_count, state)
    logger.info("End %s function", grow_shrink.__name__)
    return result
