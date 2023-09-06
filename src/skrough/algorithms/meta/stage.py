from __future__ import annotations

import logging

from attrs import define
from sklearn.base import BaseEstimator

import skrough.typing as rght
from skrough.algorithms.exceptions import LoopBreak
from skrough.algorithms.meta.aggregates import (
    ChainProcessElementsHooksAggregate,
    InnerStopHooksAggregate,
    ProcessElementsHooksAggregate,
    ProduceElementsHooksAggregate,
    StopHooksAggregate,
    UpdateStateHooksAggregate,
)
from skrough.algorithms.meta.describe import (
    autogenerate_description_node,
    describe,
    inspect_config_keys,
    inspect_input_data_keys,
    inspect_values_keys,
)
from skrough.algorithms.meta.visual_block import sk_visual_block
from skrough.logs import log_start_end
from skrough.structs.description_node import NODE_META_OPTIONAL_KEY, DescriptionNode
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@define
class Stage(rght.Describable):
    stop_agg: StopHooksAggregate
    init_agg: UpdateStateHooksAggregate
    pre_candidates_agg: ProduceElementsHooksAggregate
    candidates_agg: ProcessElementsHooksAggregate
    select_agg: ProcessElementsHooksAggregate
    filter_agg: ChainProcessElementsHooksAggregate
    inner_init_agg: ChainProcessElementsHooksAggregate
    inner_stop_agg: InnerStopHooksAggregate
    inner_process_agg: ChainProcessElementsHooksAggregate
    finalize_agg: UpdateStateHooksAggregate

    # pylint: disable-next=protected-access
    _repr_mimebundle_ = BaseEstimator._repr_mimebundle_
    _sk_visual_block_ = sk_visual_block

    @classmethod
    @log_start_end(logger)
    def from_hooks(
        cls,
        stop_hooks: rght.OneOrSequence[rght.StopHook],
        init_hooks: rght.OneOrSequence[rght.UpdateStateHook] | None,
        pre_candidates_hooks: rght.OneOrSequence[rght.ProduceElementsHook] | None,
        candidates_hooks: rght.OneOrSequence[rght.ProcessElementsHook] | None,
        select_hooks: rght.OneOrSequence[rght.ProcessElementsHook] | None,
        filter_hooks: rght.OneOrSequence[rght.ProcessElementsHook] | None,
        inner_init_hooks: rght.OneOrSequence[rght.ProcessElementsHook] | None,
        inner_stop_hooks: rght.OneOrSequence[rght.InnerStopHook],
        inner_process_hooks: rght.OneOrSequence[rght.ProcessElementsHook],
        finalize_hooks: rght.OneOrSequence[rght.UpdateStateHook] | None,
    ):
        return cls(
            stop_agg=StopHooksAggregate.from_hooks(stop_hooks),
            init_agg=UpdateStateHooksAggregate.from_hooks(init_hooks),
            pre_candidates_agg=ProduceElementsHooksAggregate.from_hooks(
                pre_candidates_hooks
            ),
            candidates_agg=ProcessElementsHooksAggregate.from_hooks(candidates_hooks),
            select_agg=ProcessElementsHooksAggregate.from_hooks(select_hooks),
            filter_agg=ChainProcessElementsHooksAggregate.from_hooks(filter_hooks),
            inner_init_agg=ChainProcessElementsHooksAggregate.from_hooks(
                inner_init_hooks
            ),
            inner_stop_agg=InnerStopHooksAggregate.from_hooks(inner_stop_hooks),
            inner_process_agg=ChainProcessElementsHooksAggregate.from_hooks(
                inner_process_hooks
            ),
            finalize_agg=UpdateStateHooksAggregate.from_hooks(finalize_hooks),
        )

    @log_start_end(logger)
    def __call__(self, state: ProcessingState) -> None:
        logger.debug("Run init hooks")
        self.init_agg(state)

        try:
            logger.debug("Check stop_hooks on start")
            self.stop_agg(state, raise_loop_break=True)

            while True:
                logger.debug("Run pre_candidates_hooks")
                pre_candidates = self.pre_candidates_agg(state)

                logger.debug("Run candidates_hooks")
                candidates = self.candidates_agg(state, pre_candidates)

                logger.debug("Run select_hooks")
                selected = self.select_agg(state, candidates)

                logger.debug("Run verify_hooks")
                filtered = self.filter_agg(state, selected)

                logger.debug("Run inner_init_hooks")
                elements = self.inner_init_agg(state, filtered)

                should_check_stop_after = True

                while True:
                    logger.debug("Check inner_stop_hooks")
                    if self.inner_stop_agg(state, elements, raise_loop_break=False):
                        logger.debug("Break inner loop")
                        break

                    logger.debug("Run inner_process_hooks")
                    elements = self.inner_process_agg(state, elements)

                    logger.debug("Check stop_hooks in inner loop")
                    self.stop_agg(state, raise_loop_break=True)
                    should_check_stop_after = False

                if should_check_stop_after:
                    logger.debug("Check stop_hooks on inner loop exit")
                    self.stop_agg(state, raise_loop_break=True)

        except LoopBreak:
            logger.debug("Break outer loop")

        logger.debug("Run finalize_hooks")
        self.finalize_agg(state)

    def get_description_graph(self):
        result = autogenerate_description_node(
            processing_element=self, process_docstring=True
        )
        result.children = [
            describe(
                self.init_agg,
                override_node_name="init",
            ),
            describe(
                self.stop_agg,
                override_node_name="check_stop",
            ),
            DescriptionNode(
                node_name="outer_loop",
                children=[
                    describe(
                        self.pre_candidates_agg,
                        override_node_name="pre_candidates",
                    ),
                    describe(
                        self.candidates_agg,
                        override_node_name="candidates",
                    ),
                    describe(
                        self.select_agg,
                        override_node_name="select",
                    ),
                    describe(
                        self.filter_agg,
                        override_node_name="filter",
                    ),
                    describe(
                        self.inner_init_agg,
                        override_node_name="inner_init",
                    ),
                    DescriptionNode(
                        node_name="inner_loop",
                        children=[
                            describe(
                                self.inner_stop_agg,
                                override_node_name="inner_check_stop",
                            ),
                            describe(
                                self.inner_process_agg,
                                override_node_name="inner_process",
                            ),
                            describe(
                                self.stop_agg,
                                override_node_name="check_stop",
                            ),
                        ],
                    ),
                    describe(
                        self.stop_agg,
                        override_node_name="check_stop",
                        override_node_meta={NODE_META_OPTIONAL_KEY: True},
                    ),
                ],
            ),
            describe(
                self.finalize_agg,
                override_node_name="finalize",
            ),
        ]
        return result

    def _get_children_processing_elements(self):
        return [
            self.stop_agg,
            self.init_agg,
            self.pre_candidates_agg,
            self.candidates_agg,
            self.select_agg,
            self.filter_agg,
            self.inner_init_agg,
            self.inner_stop_agg,
            self.inner_process_agg,
            self.finalize_agg,
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
