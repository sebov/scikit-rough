# pylint: disable=duplicate-code

import logging
from typing import Any, List, Optional, Sequence

import numpy as np
from attrs import define
from sklearn.base import BaseEstimator

import skrough.typing as rght
from skrough.algorithms.meta.aggregates import UpdateStateHooksAggregate
from skrough.algorithms.meta.describe import (
    autogenerate_description_node,
    describe,
    determine_config_keys,
    determine_input_keys,
    determine_values_keys,
)
from skrough.algorithms.meta.helpers import normalize_sequence
from skrough.algorithms.meta.stage import Stage
from skrough.algorithms.meta.visual_block import sk_visual_block
from skrough.logs import log_start_end
from skrough.structs.description_node import NODE_META_OPTIONAL_KEY
from skrough.structs.state import ProcessingState, StateConfig, StateInputData

logger = logging.getLogger(__name__)


@define
class ProcessingMultiStage(rght.Describable):
    init_multi_stage_agg: UpdateStateHooksAggregate
    init_agg: UpdateStateHooksAggregate
    stages: Sequence[Stage]
    finalize_agg: UpdateStateHooksAggregate
    prepare_result_fun: rght.PrepareResultFunction

    # pylint: disable-next=protected-access
    _repr_mimebundle_ = BaseEstimator._repr_mimebundle_
    _sk_visual_block_ = sk_visual_block

    @classmethod
    @log_start_end(logger)
    def from_hooks(
        cls,
        prepare_result_fun: rght.PrepareResultFunction,
        init_multi_stage_hooks: Optional[
            rght.OneOrSequence[rght.UpdateStateHook]
        ] = None,
        init_hooks: Optional[rght.OneOrSequence[rght.UpdateStateHook]] = None,
        stages: Optional[rght.OneOrSequence[Stage]] = None,
        finalize_hooks: Optional[rght.OneOrSequence[rght.UpdateStateHook]] = None,
    ):
        return cls(
            init_multi_stage_agg=UpdateStateHooksAggregate.from_hooks(
                init_multi_stage_hooks
            ),
            init_agg=UpdateStateHooksAggregate.from_hooks(init_hooks),
            stages=normalize_sequence(stages, optional=True),
            finalize_agg=UpdateStateHooksAggregate.from_hooks(finalize_hooks),
            prepare_result_fun=prepare_result_fun,
        )

    @log_start_end(logger)
    def __call__(
        self,
        state: Optional[ProcessingState] = None,
        input_data: Optional[StateInputData] = None,
        config: Optional[StateConfig] = None,
        seed: rght.Seed = None,
    ) -> Any:
        logger.debug("Create state object")
        if state is None:
            logger.debug("No state passed, create new one from config, input and seed")
            state = ProcessingState.from_optional(
                rng=np.random.default_rng(seed),
                processing_fun=self,
                config=config,
                input_data=input_data,
            )
            logger.debug("Run init state hooks")
            self.init_multi_stage_agg(state)

        logger.debug("Run init hooks")
        self.init_agg(state)

        logger.debug("Run stages sequentially")
        for i, stage in enumerate(self.stages):
            logger.debug("Run stage %d", i)
            stage(state)

        logger.debug("Run finalize hooks")
        self.finalize_agg(state)

        logger.debug("Prepare result function")
        result = self.prepare_result_fun(state)
        return result

    def get_description_graph(self):
        result = autogenerate_description_node(
            processing_element=self, process_docstring=True
        )
        result.children = [
            describe(
                self.init_multi_stage_agg,
                override_node_name="init_multi_stage",
                override_node_meta={NODE_META_OPTIONAL_KEY: True},
            ),
            describe(
                self.init_agg,
                override_node_name="init",
            ),
            describe(
                self.stages,
                override_node_name="stages",
            ),
            describe(
                self.finalize_agg,
                override_node_name="finalize",
            ),
            describe(
                self.prepare_result_fun,
                override_node_name="prepare_result",
            ),
        ]
        return result

    def _get_children_processing_elements(self):
        return [
            self.init_multi_stage_agg,
            self.init_agg,
            *self.stages,
            self.finalize_agg,
            self.prepare_result_fun,
        ]

    def get_config_keys(self) -> List[str]:
        return self._get_keys_from_elements(
            children=self._get_children_processing_elements(),
            determine_keys_function=determine_config_keys,
        )

    def get_input_keys(self) -> List[str]:
        return self._get_keys_from_elements(
            children=self._get_children_processing_elements(),
            determine_keys_function=determine_input_keys,
        )

    def get_values_keys(self) -> List[str]:
        return self._get_keys_from_elements(
            children=self._get_children_processing_elements(),
            determine_keys_function=determine_values_keys,
        )
