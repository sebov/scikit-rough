# pylint: disable=duplicate-code

import logging
from typing import Optional, Sequence

import docstring_parser
import numpy as np
from attrs import define
from sklearn.base import BaseEstimator

import skrough.typing as rght
from skrough.algorithms.meta.aggregates import UpdateStateHooksAggregate
from skrough.algorithms.meta.describe import NODE_META_OPTIONAL_KEY, DescriptionNode
from skrough.algorithms.meta.describe import describe as describe_fun
from skrough.algorithms.meta.helpers import normalize_hook_sequence
from skrough.algorithms.meta.stage import Stage
from skrough.algorithms.meta.visual_block import sk_visual_block
from skrough.logs import log_start_end
from skrough.structs.state import ProcessingState, StateConfig, StateInputData

logger = logging.getLogger(__name__)


@define
class ProcessingMultiStage:
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
            stages=normalize_hook_sequence(stages, optional=True),
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
    ) -> None:
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

    def describe(self):
        docstring = docstring_parser.parse(self.__doc__ or "")
        return DescriptionNode(
            name=self.__class__.__name__,
            short_description=docstring.short_description,
            long_description=docstring.long_description,
            children=[
                describe_fun(
                    self.init_multi_stage_agg,
                    override_node_name="init_multi_stage",
                    override_node_meta={NODE_META_OPTIONAL_KEY: True},
                ),
                describe_fun(
                    self.init_agg,
                    override_node_name="init",
                ),
                describe_fun(
                    self.stages,
                    override_node_name="stages",
                ),
                describe_fun(
                    self.finalize_agg,
                    override_node_name="finalize",
                ),
                describe_fun(
                    self.prepare_result_fun,
                    override_node_name="prepare_result",
                ),
            ],
        )
