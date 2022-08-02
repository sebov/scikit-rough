import logging
from typing import Any, Callable, List, Optional, Sequence, TypeVar

import pandas as pd

import skrough.typing as rght
from skrough.logs import log_start_end
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=Callable)


@log_start_end(logger)
def normalize_hook_sequence(
    hooks: Optional[rght.OneOrSequence[T]],
    optional: bool,
) -> Sequence[T]:
    if optional is False and not hooks:
        raise ValueError("Hooks argument should not be empty.")
    result: Sequence[T]
    if hooks is None:
        result = []
    elif not isinstance(hooks, Sequence):
        result = [hooks]
    else:
        result = hooks
    return result


@log_start_end(logger)
def aggregate_update_state_hooks(
    hooks: Optional[rght.OneOrSequence[rght.UpdateStateHook]],
) -> rght.UpdateStateHook:
    normalized_hooks = normalize_hook_sequence(hooks, optional=True)

    @log_start_end(logger)
    def _update_state_function(
        state: ProcessingState,
    ) -> None:
        for hook in normalized_hooks:
            hook(state)

    return _update_state_function


@log_start_end(logger)
def aggregate_produce_elements_hooks(
    hooks: Optional[rght.OneOrSequence[rght.ProduceElementsHook]],
) -> rght.ProduceElementsFunction:
    normalized_hooks = normalize_hook_sequence(hooks, optional=True)

    @log_start_end(logger)
    def _produce_elements_function(
        state: ProcessingState,
    ) -> rght.Elements:
        result: List[Any] = []
        for hook in normalized_hooks:
            result.extend(hook(state))
        return pd.unique(result)

    return _produce_elements_function


@log_start_end(logger)
def aggregate_process_elements_hooks(
    hooks: Optional[rght.OneOrSequence[rght.ProcessElementsHook]],
) -> rght.ProcessElementsFunction:
    normalized_hooks = normalize_hook_sequence(hooks, optional=True)

    @log_start_end(logger)
    def _process_elements_function(
        state: ProcessingState,
        elements: rght.Elements,
    ) -> rght.Elements:
        result: List[Any] = []
        for hook in normalized_hooks:
            result.extend(hook(state, elements))
        return pd.unique(result)

    return _process_elements_function


@log_start_end(logger)
def aggregate_chain_process_elements_hooks(
    hooks: Optional[rght.OneOrSequence[rght.ProcessElementsHook]],
) -> rght.ProcessElementsFunction:
    normalized_hooks = normalize_hook_sequence(hooks, optional=True)

    @log_start_end(logger)
    def _process_elements_function(
        state: ProcessingState,
        elements: rght.Elements,
    ) -> rght.Elements:
        result = elements
        for hook in normalized_hooks:
            result = hook(state, result)
        return result

    return _process_elements_function
