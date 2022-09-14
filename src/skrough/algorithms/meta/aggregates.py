import logging
from typing import Any, List, Optional

import docstring_parser
import pandas as pd
from attrs import define

import skrough.typing as rght
from skrough.algorithms.exceptions import LoopBreak
from skrough.algorithms.meta.describe import DescriptionNode
from skrough.algorithms.meta.describe import describe as describe_fun
from skrough.algorithms.meta.helpers import normalize_hook_sequence
from skrough.logs import log_start_end
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


class AggregateMixin:
    def describe(self):
        docstring = docstring_parser.parse(self.__doc__ or "")
        short_description = docstring.short_description
        long_description = docstring.long_description

        hooks_list_description = describe_fun(self.normalized_hooks)  # type: ignore

        return DescriptionNode(
            name=self.__class__.__name__,
            short_description=short_description,
            long_description=long_description,
            children=hooks_list_description.children,
        )


@define
class StopHooksAggregate(AggregateMixin):
    normalized_hooks: List[rght.StopHook]

    @classmethod
    @log_start_end(logger)
    def from_hooks(
        cls,
        hooks: rght.OneOrSequence[rght.StopHook],
    ):
        normalized_hooks = normalize_hook_sequence(hooks, optional=False)
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


@define
class InnerStopHooksAggregate(AggregateMixin):
    normalized_hooks: List[rght.InnerStopHook]

    @classmethod
    @log_start_end(logger)
    def from_hooks(
        cls,
        hooks: rght.OneOrSequence[rght.InnerStopHook],
    ):
        normalized_hooks = normalize_hook_sequence(hooks, optional=False)
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


@define
class UpdateStateHooksAggregate(AggregateMixin):
    normalized_hooks: List[rght.UpdateStateHook]

    @classmethod
    @log_start_end(logger)
    def from_hooks(
        cls,
        hooks: Optional[rght.OneOrSequence[rght.UpdateStateHook]],
    ):
        normalized_hooks = normalize_hook_sequence(hooks, optional=True)
        return cls(normalized_hooks=normalized_hooks)

    @log_start_end(logger)
    def __call__(
        self,
        state: ProcessingState,
    ) -> None:
        for hook in self.normalized_hooks:
            hook(state)


@define
class ProduceElementsHooksAggregate(AggregateMixin):
    normalized_hooks: List[rght.ProduceElementsHook]

    @classmethod
    @log_start_end(logger)
    def from_hooks(
        cls,
        hooks: Optional[rght.OneOrSequence[rght.ProduceElementsHook]],
    ):
        normalized_hooks = normalize_hook_sequence(hooks, optional=True)
        return cls(normalized_hooks=normalized_hooks)

    @log_start_end(logger)
    def __call__(
        self,
        state: ProcessingState,
    ) -> rght.Elements:
        result: List[Any] = []
        for hook in self.normalized_hooks:
            result.extend(hook(state))
        return pd.unique(result)


@define
class ProcessElementsHooksAggregate(AggregateMixin):
    normalized_hooks: List[rght.ProcessElementsHook]

    @classmethod
    @log_start_end(logger)
    def from_hooks(
        cls,
        hooks: Optional[rght.OneOrSequence[rght.ProcessElementsHook]],
    ):
        normalized_hooks = normalize_hook_sequence(hooks, optional=True)
        return cls(normalized_hooks=normalized_hooks)

    @log_start_end(logger)
    def __call__(
        self,
        state: ProcessingState,
        elements: rght.Elements,
    ) -> rght.Elements:
        result: List[Any] = []
        for hook in self.normalized_hooks:
            result.extend(hook(state, elements))
        return pd.unique(result)


@define
class ChainProcessElementsHooksAggregate(AggregateMixin):
    normalized_hooks: List[rght.ProcessElementsHook]

    @classmethod
    @log_start_end(logger)
    def from_hooks(
        cls,
        hooks: Optional[rght.OneOrSequence[rght.ProcessElementsHook]],
    ):
        normalized_hooks = normalize_hook_sequence(hooks, optional=True)
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
