import itertools
import logging
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

import skrough.typing as rght
from skrough.algorithms.hooks.names import RESULT_ATTRS, SINGLE_GROUP_INDEX
from skrough.algorithms.meta.exceptions import LoopBreak
from skrough.algorithms.meta.helpers import (
    aggregate_grow_stop_hooks,
    aggregate_shrink_accept_hooks,
    normalize_hook_sequence,
    run_update_hooks,
)
from skrough.logs import log_start_end
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import GrowShrinkState, StateConfig

logger = logging.getLogger(__name__)


@log_start_end(logger)
def grow_shrink(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    config: StateConfig,
    init_hooks: Optional[
        Union[
            rght.GSUpdateStateHook,
            Sequence[rght.GSUpdateStateHook],
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
            rght.GSUpdateStateHook,
            Sequence[rght.GSUpdateStateHook],
        ]
    ],
    prepare_result_hook: rght.GSPrepareResultHook,
    seed: rght.Seed = None,
):
    logger.debug("Create state object")
    rng = np.random.default_rng(seed)
    state = GrowShrinkState(
        rng=rng,
        config=config,
    )

    logger.debug("Normalize init_hooks")
    init_hooks = normalize_hook_sequence(
        init_hooks,
        optional=True,
    )

    logger.debug("Normalize grow_stop_hooks")
    grow_stop_hooks = normalize_hook_sequence(
        grow_stop_hooks,
        optional=False,
    )

    logger.debug("Normalize get_candidate_attrs_hooks")
    grow_candidate_attrs_hooks = normalize_hook_sequence(
        grow_candidate_attrs_hooks,
        optional=True,
    )

    logger.debug("Normalize post_select_attrs_hooks")
    grow_post_select_attrs_hooks = normalize_hook_sequence(
        grow_post_select_attrs_hooks,
        optional=True,
    )

    logger.debug("Normalize post_select_attrs_hooks")
    shrink_accept_group_index_hooks = normalize_hook_sequence(
        shrink_accept_group_index_hooks,
        optional=True,
    )

    logger.debug("Normalize finalize_hooks")
    finalize_hooks = normalize_hook_sequence(
        finalize_hooks,
        optional=True,
    )

    grow_stop_check = aggregate_grow_stop_hooks(grow_stop_hooks)

    shrink_accept_group_index_check = aggregate_shrink_accept_hooks(
        shrink_accept_group_index_hooks
    )

    logger.debug("Run init hooks")
    run_update_hooks(x, x_counts, y, y_count, state, init_hooks)

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
                state.values[RESULT_ATTRS],
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
                    state.values[RESULT_ATTRS].append(selected_attr)
                    state.values[SINGLE_GROUP_INDEX] = state.values[
                        SINGLE_GROUP_INDEX
                    ].split(
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

    logger.info("Attrs after grow phase = %s", state.values[RESULT_ATTRS])

    ##################
    # shrink phase
    ##################
    logger.info("Start shrink phase")

    if shrink_candidate_attrs_hook is None:
        shrink_candidate_attrs = np.asarray(list(reversed(state.values[RESULT_ATTRS])))
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
        shrinked_attrs = state.values[RESULT_ATTRS][:]
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
            state.values[RESULT_ATTRS] = shrinked_attrs
            state.values[SINGLE_GROUP_INDEX] = shrinked_group_index

    logger.info("End shrink phase")
    ##################
    # end shrink phase
    ##################

    logger.debug("Run finalize hooks")
    run_update_hooks(x, x_counts, y, y_count, state, finalize_hooks)

    result = prepare_result_hook(x, x_counts, y, y_count, state)
    return result
