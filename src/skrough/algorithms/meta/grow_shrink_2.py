import logging

import numpy as np

import skrough.typing as rght
from skrough.logs import log_start_end
from skrough.structs.state import GrowShrinkState, StateConfig, StateInput

logger = logging.getLogger(__name__)


@log_start_end(logger)
def grow_shrink_2(
    input: StateInput,
    config: StateConfig,
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
    prepare_result_hook: rght.PrepareResultHook,
    seed: rght.Seed = None,
):

    logger.debug("Create state object")
    rng = np.random.default_rng(seed)
    state = GrowShrinkState(
        rng=rng,
        config=config,
        input=input,
    )

    result = prepare_result_hook(state)
    return result
