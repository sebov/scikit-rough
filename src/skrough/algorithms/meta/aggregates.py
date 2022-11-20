import itertools
import logging
from typing import Any, Callable, List, Optional

import docstring_parser
import pandas as pd
from attrs import define
from sklearn.base import BaseEstimator

import skrough.typing as rght
from skrough.algorithms.exceptions import LoopBreak
from skrough.algorithms.meta.describe import (
    DescriptionNode,
    describe,
    determine_config_keys,
    determine_input_keys,
    determine_values_keys,
)
from skrough.algorithms.meta.helpers import normalize_sequence
from skrough.algorithms.meta.visual_block import sk_visual_block
from skrough.logs import log_start_end
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


class AggregateMixin(rght.Describable):

    # pylint: disable-next=protected-access
    _repr_mimebundle_ = BaseEstimator._repr_mimebundle_
    _sk_visual_block_ = sk_visual_block

    def get_description_graph(self):
        """Return the description of the processing element."""
        docstring = docstring_parser.parse(self.__doc__ or "")
        short_description = docstring.short_description
        long_description = docstring.long_description

        config_keys = self.get_config_keys()
        input_keys = self.get_input_keys()
        values_keys = self.get_values_keys()

        hooks_list_description = describe(self.normalized_hooks)  # type: ignore

        return DescriptionNode(
            name=self.__class__.__name__,
            short_description=short_description,
            long_description=long_description,
            config_keys=config_keys,
            input_keys=input_keys,
            values_keys=values_keys,
            children=hooks_list_description.children,
        )

    def _get_keys(self, determine_keys_function: Callable) -> List[str]:
        return list(
            set(
                itertools.chain.from_iterable(
                    [
                        determine_keys_function(child)
                        for child in self.normalized_hooks  # type: ignore
                    ],
                )
            )
        )

    def get_config_keys(self) -> List[str]:
        return self._get_keys(determine_config_keys)

    def get_input_keys(self) -> List[str]:
        return self._get_keys(determine_input_keys)

    def get_values_keys(self) -> List[str]:
        return self._get_keys(determine_values_keys)


@define
class StopHooksAggregate(AggregateMixin):
    normalized_hooks: List[rght.StopHook]

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


@define
class InnerStopHooksAggregate(AggregateMixin):
    normalized_hooks: List[rght.InnerStopHook]

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


@define
class UpdateStateHooksAggregate(AggregateMixin):
    normalized_hooks: List[rght.UpdateStateHook]

    @classmethod
    @log_start_end(logger)
    def from_hooks(
        cls,
        hooks: Optional[rght.OneOrSequence[rght.UpdateStateHook]],
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


@define
class ProduceElementsHooksAggregate(AggregateMixin):
    normalized_hooks: List[rght.ProduceElementsHook]

    @classmethod
    @log_start_end(logger)
    def from_hooks(
        cls,
        hooks: Optional[rght.OneOrSequence[rght.ProduceElementsHook]],
    ):
        normalized_hooks = normalize_sequence(hooks, optional=True)
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
        normalized_hooks = normalize_sequence(hooks, optional=True)
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
