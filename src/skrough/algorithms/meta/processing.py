import logging
from typing import Optional, Sequence

import numpy as np
from attrs import define

import skrough.typing as rght
from skrough.algorithms.meta.helpers import (
    aggregate_update_state_hooks,
    normalize_hook_sequence,
)
from skrough.algorithms.meta.stage import ProcessingStage
from skrough.logs import log_start_end
from skrough.structs.state import ProcessingState, StateConfig, StateInputData

logger = logging.getLogger(__name__)


# @log_start_end(logger)
# def process_multi_stage(
#     input: StateInput,
#     config: StateConfig,
#     init_hooks: Optional[rght.OneOrSequence[rght.UpdateStateHook]],
#     process_stages: rght.OneOrSequence[ProcessingStage],
#     finalize_hooks: Optional[rght.OneOrSequence[rght.UpdateStateHook]],
#     prepare_result_fun: rght.PrepareResultFunction,
#     seed: rght.Seed = None,
# ):
#     init_fun = aggregate_update_state_hooks(init_hooks)
#     process_stages = normalize_hook_sequence(process_stages, optional=False)
#     finalize_fun = aggregate_update_state_hooks(finalize_hooks)

#     logger.debug("Create state object")
#     rng = np.random.default_rng(seed)
#     state = ProcessingState(
#         rng=rng,
#         config=config,
#         input_data=input,
#         processing_fun=lambda x: None,  # dummy function
#     )

#     logger.debug("Run grow_shrink init_hooks")
#     init_fun(state)

#     for stage in process_stages:
#         stage(state)

#     logger.debug("Run grow_shrink finalize_hooks")
#     finalize_fun(state)

#     result = prepare_result_fun(state)
#     return result


@define
class ProcessingMultiStage:
    init_state_fun: rght.UpdateStateFunction
    init_fun: rght.UpdateStateFunction
    stages: Sequence[ProcessingStage]
    finalize_fun: rght.UpdateStateFunction
    prepare_result_fun: rght.PrepareResultFunction

    @classmethod
    @log_start_end(logger)
    def from_hooks(
        cls,
        prepare_result_fun: rght.PrepareResultFunction,
        init_state_hooks: Optional[rght.OneOrSequence[rght.UpdateStateHook]] = None,
        init_hooks: Optional[rght.OneOrSequence[rght.UpdateStateHook]] = None,
        process_stages: Optional[rght.OneOrSequence[ProcessingStage]] = None,
        finalize_hooks: Optional[rght.OneOrSequence[rght.UpdateStateHook]] = None,
    ):
        return cls(
            init_state_fun=aggregate_update_state_hooks(init_state_hooks),
            init_fun=aggregate_update_state_hooks(init_hooks),
            stages=normalize_hook_sequence(process_stages, optional=True),
            finalize_fun=aggregate_update_state_hooks(finalize_hooks),
            prepare_result_fun=prepare_result_fun,
        )

    @log_start_end(logger)
    def __call__(
        self,
        state: Optional[ProcessingState] = None,
        input: Optional[StateInputData] = None,
        config: Optional[StateConfig] = None,
        seed: rght.Seed = None,
    ) -> None:
        logger.debug("Create state object")
        if state is None:
            logger.debug("No state passed, create new one from config, input and seed")
            state = ProcessingState.create_from_optional(
                rng=np.random.default_rng(seed),
                processing_fun=self,
                config=config,
                input_data=input,
            )
            logger.debug("Run init state hooks")
            self.init_state_fun(state)

        logger.debug("Run init hooks")
        self.init_fun(state)

        logger.debug("Run stages sequentially")
        for i, stage in enumerate(self.stages):
            logger.debug("Run stage %d", 0)
            stage(state)

        logger.debug("Run finalize hooks")
        self.finalize_fun(state)

        logger.debug("Prepare result function")
        result = self.prepare_result_fun(state)
        return result
