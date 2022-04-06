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
    check_stop_hooks: Union[
        rght.GSCheckStopHook,
        Sequence[rght.GSCheckStopHook],
    ],
    get_candidate_attrs_hooks: Optional[
        Union[
            rght.GSGetCandidateAttrsHook,
            Sequence[rght.GSGetCandidateAttrsHook],
        ]
    ],
    select_attrs_hook: rght.GSSelectAttrsHook,
    post_select_attrs_hooks: Optional[
        Union[
            rght.GSPostSelectAttrsHook,
            Sequence[rght.GSPostSelectAttrsHook],
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

    logger.debug("Normalize check_stop_hooks")
    if not isinstance(check_stop_hooks, Sequence):
        check_stop_hooks = [check_stop_hooks]

    logger.debug("Normalize get_candidate_attrs_hooks")
    if (get_candidate_attrs_hooks is not None) and (
        not isinstance(get_candidate_attrs_hooks, Sequence)
    ):
        get_candidate_attrs_hooks = [get_candidate_attrs_hooks]

    logger.debug("Normalize post_select_attrs_hooks")
    if (post_select_attrs_hooks is not None) and (
        not isinstance(post_select_attrs_hooks, Sequence)
    ):
        post_select_attrs_hooks = [post_select_attrs_hooks]

    logger.debug("Normalize finalize_hooks")
    if (finalize_hooks is not None) and (not isinstance(finalize_hooks, Sequence)):
        finalize_hooks = [finalize_hooks]

    def check_stop(check_stop_hooks: Sequence[rght.GSCheckStopHook]):
        # stop hooks
        if any(
            check_stop_hook(x, x_counts, y, y_count, state)
            for check_stop_hook in check_stop_hooks
        ):
            raise LoopBreak()

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
        check_stop(check_stop_hooks)

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
            if get_candidate_attrs_hooks is None:
                logger.debug("Use all remaining attrs as candidate attrs")
                candidate_attrs = remaining_attrs
            else:
                logger.debug("Obtain candidate attrs using candidate attrs hooks")
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
                logger.debug("Remove duplicates from candidate attrs")
                candidate_attrs = pd.unique(candidate_attrs)

            # select attrs hook
            logger.debug("Select attrs using select attrs hooks")
            selected_attrs = select_attrs_hook(
                x,
                x_counts,
                y,
                y_count,
                state,
                candidate_attrs,
            )
            logger.info("Selected attrs = %s", selected_attrs)

            logger.debug("Run post select hooks")
            if post_select_attrs_hooks is not None:
                for post_select_attrs_hook in post_select_attrs_hooks:
                    selected_attrs = post_select_attrs_hook(
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
                check_stop(check_stop_hooks)
            else:
                logger.debug("Add selected attrs one by one")
                for selected_attr in selected_attrs:
                    logger.info("Add attr <%d>", selected_attr)
                    state.group_index = state.group_index.split(
                        x[:, selected_attr],
                        x_counts[selected_attr],
                    )
                    state.result_attrs.append(  # pylint: disable=no-member
                        selected_attr
                    )
                    logger.debug("Check stop conditions")
                    check_stop(check_stop_hooks)

    except LoopBreak:
        pass
    logger.info("End grow phase")
    ################
    # end grow phase
    ################

    ##################
    # shrink phase
    ##################
    logger.info("Start shrink phase")
    # ...

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
