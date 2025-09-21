import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

import skrough.typing as rght
from skrough.algorithms.exceptions import LoopBreak
from skrough.algorithms.meta.describe import (
    autogenerate_description_node,
    describe,
)
from skrough.algorithms.meta.helpers import normalize_sequence
from skrough.logs import log_start_end
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


class AggregateMixin(rght.Describable):
    def get_description_graph(self):
        """Return the description of an aggregate processing element."""
        result = autogenerate_description_node(
            processing_element=self, process_docstring=True
        )
        hooks_list_description = describe(self.normalized_hooks)  # type: ignore
        result.children = hooks_list_description.children
        return result


@dataclass
class StopHooksAggregate(AggregateMixin):
    normalized_hooks: list[rght.StopHook]

    @classmethod
    @log_start_end(logger)
    def from_hooks(
        cls,
        hooks: rght.OneOrSequence[rght.StopHook],
    ):
        normalized_hooks = normalize_sequence(hooks, optional=False)
        return cls(normalized_hooks=normalized_hooks)

    @log_start_end(logger)
    def __call__(
        self,
        state: ProcessingState,
        raise_loop_break: bool,
    ) -> bool:
        result = any(stop_hook(state) for stop_hook in self.normalized_hooks)
        if result and raise_loop_break:
            raise LoopBreak()
        return result


@dataclass
class InnerStopHooksAggregate(AggregateMixin):
    normalized_hooks: list[rght.InnerStopHook]

    @classmethod
    @log_start_end(logger)
    def from_hooks(
        cls,
        hooks: rght.OneOrSequence[rght.InnerStopHook],
    ):
        normalized_hooks = normalize_sequence(hooks, optional=False)
        return cls(normalized_hooks=normalized_hooks)

    @log_start_end(logger)
    def __call__(
        self,
        state: ProcessingState,
        elements: rght.Elements,
        raise_loop_break: bool,
    ) -> bool:
        result = any(
            stop_hook(state=state, elements=elements)
            for stop_hook in self.normalized_hooks
        )
        if result and raise_loop_break:
            raise LoopBreak()
        return result


@dataclass
class UpdateStateHooksAggregate(AggregateMixin):
    normalized_hooks: list[rght.UpdateStateHook]

    @classmethod
    @log_start_end(logger)
    def from_hooks(
        cls,
        hooks: rght.OneOrSequence[rght.UpdateStateHook] | None,
    ):
        normalized_hooks = normalize_sequence(hooks, optional=True)
        return cls(normalized_hooks=normalized_hooks)

    @log_start_end(logger)
    def __call__(
        self,
        state: ProcessingState,
    ) -> None:
        for hook in self.normalized_hooks:
            hook(state)


@dataclass
class ProduceElementsHooksAggregate(AggregateMixin):
    normalized_hooks: list[rght.ProduceElementsHook]

    @classmethod
    @log_start_end(logger)
    def from_hooks(
        cls,
        hooks: rght.OneOrSequence[rght.ProduceElementsHook] | None,
    ):
        normalized_hooks = normalize_sequence(hooks, optional=True)
        return cls(normalized_hooks=normalized_hooks)

    @log_start_end(logger)
    def __call__(
        self,
        state: ProcessingState,
    ) -> rght.Elements:
        result: list[Any] = []
        for hook in self.normalized_hooks:
            result.extend(hook(state))
        return pd.Series(result).unique()


@dataclass
class ProcessElementsHooksAggregate(AggregateMixin):
    normalized_hooks: list[rght.ProcessElementsHook]

    @classmethod
    @log_start_end(logger)
    def from_hooks(
        cls,
        hooks: rght.OneOrSequence[rght.ProcessElementsHook] | None,
    ):
        normalized_hooks = normalize_sequence(hooks, optional=True)
        return cls(normalized_hooks=normalized_hooks)

    @log_start_end(logger)
    def __call__(
        self,
        state: ProcessingState,
        elements: rght.Elements,
    ) -> rght.Elements:
        result: list[Any] = []
        for hook in self.normalized_hooks:
            result.extend(hook(state, elements))
        return pd.Series(result).unique()


@dataclass
class ChainProcessElementsHooksAggregate(AggregateMixin):
    normalized_hooks: list[rght.ProcessElementsHook]

    @classmethod
    @log_start_end(logger)
    def from_hooks(
        cls,
        hooks: rght.OneOrSequence[rght.ProcessElementsHook] | None,
    ):
        normalized_hooks = normalize_sequence(hooks, optional=True)
        return cls(normalized_hooks=normalized_hooks)

    @log_start_end(logger)
    def __call__(
        self,
        state: ProcessingState,
        elements: rght.Elements,
    ) -> rght.Elements:
        result = elements
        for hook in self.normalized_hooks:
            result = hook(state, result)
        return result
