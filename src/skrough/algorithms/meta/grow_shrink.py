import logging

import numpy as np

import skrough.typing as rght
from skrough.algorithms.meta.helpers import (
    aggregate_update_state_hooks,
    normalize_hook_sequence,
)
from skrough.algorithms.meta.loop import ProcessingStage, loop
from skrough.logs import log_start_end
from skrough.structs.state import GrowShrinkState, StateConfig, StateInput

logger = logging.getLogger(__name__)


@log_start_end(logger)
def grow_shrink(
    input: StateInput,
    config: StateConfig,
    init_hooks: rght.OptionalOneOrSequence[rght.UpdateStateHook],
    finalize_hooks: rght.OptionalOneOrSequence[rght.UpdateStateHook],
    prepare_result_hook: rght.PrepareResultHook,
    grow_stop_hooks: rght.OneOrSequence[rght.StopHook],
    grow_init_hooks: rght.OptionalOneOrSequence[rght.UpdateStateHook],
    grow_pre_candidates_hooks: rght.OptionalOneOrSequence[rght.ProduceElementsHook],
    grow_candidates_hooks: rght.OptionalOneOrSequence[rght.ProcessElementsHook],
    grow_select_hooks: rght.OptionalOneOrSequence[rght.ProcessElementsHook],
    grow_verify_hooks: rght.OptionalOneOrSequence[rght.ProcessElementsHook],
    grow_inner_init_hooks: rght.OptionalOneOrSequence[rght.ProcessElementsHook],
    grow_inner_stop_hooks: rght.OneOrSequence[rght.InnerStopHook],
    grow_inner_process_hooks: rght.OneOrSequence[rght.ProcessElementsHook],
    grow_finalize_hooks: rght.OptionalOneOrSequence[rght.UpdateStateHook],
    seed: rght.Seed = None,
):
    init_fun = aggregate_update_state_hooks(init_hooks)
    finalize_fun = aggregate_update_state_hooks(finalize_hooks)

    logger.debug("Create state object")
    rng = np.random.default_rng(seed)
    state = GrowShrinkState(
        rng=rng,
        config=config,
        input=input,
    )

    logger.debug("Run grow_shrink init_hooks")
    init_fun(state)

    loop(
        state,
        grow_stop_hooks,
        grow_init_hooks,
        grow_pre_candidates_hooks,
        grow_candidates_hooks,
        grow_select_hooks,
        grow_verify_hooks,
        grow_inner_init_hooks,
        grow_inner_stop_hooks,
        grow_inner_process_hooks,
        grow_finalize_hooks,
    )

    logger.debug("Run grow_shrink finalize_hooks")
    finalize_fun(state)

    result = prepare_result_hook(state)
    return result


@log_start_end(logger)
def grow_shrink_2(
    input: StateInput,
    config: StateConfig,
    init_hooks: rght.OptionalOneOrSequence[rght.UpdateStateHook],
    process_stages: rght.OneOrSequence[ProcessingStage],
    finalize_hooks: rght.OptionalOneOrSequence[rght.UpdateStateHook],
    prepare_result_hook: rght.PrepareResultHook,
    seed: rght.Seed = None,
):
    init_fun = aggregate_update_state_hooks(init_hooks)
    process_stages = normalize_hook_sequence(process_stages, optional=False)
    finalize_fun = aggregate_update_state_hooks(finalize_hooks)

    logger.debug("Create state object")
    rng = np.random.default_rng(seed)
    state = GrowShrinkState(
        rng=rng,
        config=config,
        input=input,
    )

    logger.debug("Run grow_shrink init_hooks")
    init_fun(state)

    for stage in process_stages:
        stage(state)

    logger.debug("Run grow_shrink finalize_hooks")
    finalize_fun(state)

    result = prepare_result_hook(state)
    return result
