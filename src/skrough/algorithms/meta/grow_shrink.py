import logging

import numpy as np

import skrough.typing as rght
from skrough.algorithms.meta.helpers import (
    aggregate_update_state_hooks,
    normalize_hook_sequence,
)
from skrough.algorithms.meta.loop import ProcessingStage
from skrough.logs import log_start_end
from skrough.structs.state import ProcessingState, StateConfig, StateInput

logger = logging.getLogger(__name__)


@log_start_end(logger)
def process_multi_stage(
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
    state = ProcessingState(
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
