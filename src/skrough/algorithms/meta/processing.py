import logging

import numpy as np

import skrough.typing as rght
from skrough.algorithms.meta.helpers import (
    aggregate_update_state_hooks,
    normalize_hook_sequence,
)
from skrough.algorithms.meta.stage import ProcessingStage
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


# @define
# class ProcessingMultiStage:
#     init_state_fun: rght.UpdateStateFunction
#     init_fun: rght.UpdateStateFunction
#     stages: Sequence[ProcessingStage]
#     finalize_fun: rght.UpdateStateFunction
#     prepare_result_hook: rght.PrepareResultHook

#     @classmethod
#     @log_start_end(logger)
#     def from_hooks(
#         cls,
#         init_state_hooks: rght.OptionalOneOrSequence[rght.UpdateStateHook],
#         init_hooks: rght.OptionalOneOrSequence[rght.UpdateStateHook],
#         process_stages: rght.OptionalOneOrSequence[ProcessingStage],
#         finalize_hooks: rght.OptionalOneOrSequence[rght.UpdateStateHook],
#         prepare_result_hook: rght.PrepareResultHook,
#     ):
#         return cls(
#             init_state_fun=aggregate_update_state_hooks(init_state_hooks),
#             init_fun=aggregate_update_state_hooks(init_hooks),
#             stages=normalize_hook_sequence(process_stages, optional=True) or [],
#             finalize_fun=aggregate_update_state_hooks(finalize_hooks),
#             prepare_result_hook=prepare_result_hook,
#         )

#     @log_start_end(logger)
#     def __call__(
#         self,
#         state: Optional[ProcessingState],
#         input: StateInput,
#         config: StateConfig,
#         seed: rght.Seed = None,
#     ) -> None:
#         logger.debug("Create state object")
#         rng = np.random.default_rng(seed)
#         state = ProcessingState(
#             rng=rng,
#             config=config,
#             input=input,
#         )

#         logger.debug("Run init hooks")
#         self.init_fun(state)

#         logger.debug("Run stages sequentially")
#         for i, stage in enumerate(self.stages):
#             logger.debug("Run stage %d", 0)
#             stage(state)

#         logger.debug("Run finalize hooks")
#         self.finalize_fun(state)
