# pylint: disable=duplicate-code

from __future__ import annotations

import logging
from typing import Any, Sequence, cast

import joblib
import numpy as np
from attrs import define
from sklearn.base import BaseEstimator

import skrough.typing as rght
from skrough.algorithms.constants import RNG_INTEGERS_PARAM
from skrough.algorithms.meta.aggregates import UpdateStateHooksAggregate
from skrough.algorithms.meta.describe import (
    autogenerate_description_node,
    describe,
    inspect_config_keys,
    inspect_input_data_keys,
    inspect_values_keys,
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
        init_multi_stage_hooks: None
        | (rght.OneOrSequence[rght.UpdateStateHook]) = None,
        init_hooks: rght.OneOrSequence[rght.UpdateStateHook] | None = None,
        stages: rght.OneOrSequence[Stage] | None = None,
        finalize_hooks: rght.OneOrSequence[rght.UpdateStateHook] | None = None,
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
        state: ProcessingState | None = None,
        input_data: StateInputData | None = None,
        config: StateConfig | None = None,
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

    @log_start_end(logger)
    def call_parallel(
        self,
        n_times: int,
        state: ProcessingState | None = None,
        input_data: StateInputData | None = None,
        config: StateConfig | None = None,
        seed: rght.Seed = None,
        n_jobs: int | None = None,
    ) -> list[Any]:
        rng = np.random.default_rng(seed)
        result = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(self)(
                state=state,
                input_data=input_data,
                config=config,
                seed=rng.integers(RNG_INTEGERS_PARAM),
            )
            for _ in range(n_times)
        )
        return cast(list[Any], result)

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

    def get_config_keys(self) -> list[str]:
        return self._get_keys_from_elements(
            children=self._get_children_processing_elements(),
            inspect_keys_function=inspect_config_keys,
        )

    def get_input_data_keys(self) -> list[str]:
        return self._get_keys_from_elements(
            children=self._get_children_processing_elements(),
            inspect_keys_function=inspect_input_data_keys,
        )

    def get_values_keys(self) -> list[str]:
        return self._get_keys_from_elements(
            children=self._get_children_processing_elements(),
            inspect_keys_function=inspect_values_keys,
        )
