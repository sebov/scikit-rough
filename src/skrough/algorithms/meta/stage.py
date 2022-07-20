import logging
from typing import Optional

from attrs import define

import skrough.typing as rght
from skrough.algorithms.exceptions import LoopBreak
from skrough.algorithms.meta.helpers import (
    aggregate_any_inner_stop_hooks,
    aggregate_any_stop_hooks,
    aggregate_chain_process_elements_hooks,
    aggregate_process_elements_hooks,
    aggregate_produce_elements_hooks,
    aggregate_update_state_hooks,
)
from skrough.logs import log_start_end
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@define
class ProcessingStage:
    stop_fun: rght.StopFunction
    init_fun: rght.UpdateStateFunction
    pre_candidates_fun: rght.ProduceElementsFunction
    candidates_fun: rght.ProcessElementsFunction
    select_fun: rght.ProcessElementsFunction
    filter_fun: rght.ProcessElementsFunction
    inner_init_fun: rght.ProcessElementsFunction
    inner_stop_fun: rght.InnerStopFunction
    inner_process_fun: rght.ProcessElementsFunction
    finalize_fun: rght.UpdateStateFunction

    @classmethod
    @log_start_end(logger)
    def from_hooks(
        cls,
        stop_hooks: rght.OneOrSequence[rght.StopHook],
        init_hooks: Optional[rght.OneOrSequence[rght.UpdateStateHook]],
        pre_candidates_hooks: Optional[rght.OneOrSequence[rght.ProduceElementsHook]],
        candidates_hooks: Optional[rght.OneOrSequence[rght.ProcessElementsHook]],
        select_hooks: Optional[rght.OneOrSequence[rght.ProcessElementsHook]],
        filter_hooks: Optional[rght.OneOrSequence[rght.ProcessElementsHook]],
        inner_init_hooks: Optional[rght.OneOrSequence[rght.ProcessElementsHook]],
        inner_stop_hooks: rght.OneOrSequence[rght.InnerStopHook],
        inner_process_hooks: rght.OneOrSequence[rght.ProcessElementsHook],
        finalize_hooks: Optional[rght.OneOrSequence[rght.UpdateStateHook]],
    ):
        return cls(
            stop_fun=aggregate_any_stop_hooks(stop_hooks),
            init_fun=aggregate_update_state_hooks(init_hooks),
            pre_candidates_fun=aggregate_produce_elements_hooks(pre_candidates_hooks),
            candidates_fun=aggregate_process_elements_hooks(candidates_hooks),
            select_fun=aggregate_process_elements_hooks(select_hooks),
            filter_fun=aggregate_chain_process_elements_hooks(filter_hooks),
            inner_init_fun=aggregate_chain_process_elements_hooks(inner_init_hooks),
            inner_stop_fun=aggregate_any_inner_stop_hooks(inner_stop_hooks),
            inner_process_fun=aggregate_chain_process_elements_hooks(
                inner_process_hooks
            ),
            finalize_fun=aggregate_update_state_hooks(finalize_hooks),
        )

    @log_start_end(logger)
    def __call__(self, state: ProcessingState) -> None:
        logger.debug("Run init hooks")
        self.init_fun(state)

        try:

            logger.debug("Check stop_hooks on start")
            self.stop_fun(state, raise_exception=True)

            while True:

                logger.debug("Run pre_candidates_hooks")
                pre_candidates = self.pre_candidates_fun(state)

                logger.debug("Run candidates_hooks")
                candidates = self.candidates_fun(state, pre_candidates)

                logger.debug("Run select_hooks")
                selected = self.select_fun(state, candidates)

                logger.debug("Run verify_hooks")
                filtered = self.filter_fun(state, selected)

                logger.debug("Run inner_init_hooks")
                elements = self.inner_init_fun(state, filtered)

                should_check_stop_after = True

                while True:

                    logger.debug("Check inner_stop_hooks")
                    if self.inner_stop_fun(state, elements):
                        logger.debug("Break inner loop")
                        break

                    logger.debug("Run inner_process_hooks")
                    elements = self.inner_process_fun(state, elements)

                    logger.debug("Check stop_hooks in inner loop")
                    self.stop_fun(state, raise_exception=True)
                    should_check_stop_after = False

                if should_check_stop_after:
                    logger.debug("Check stop_hooks on inner loop exit")
                    self.stop_fun(state, raise_exception=True)

        except LoopBreak:
            logger.debug("Break outer loop")

        logger.debug("Run finalize_hooks")
        self.finalize_fun(state)
