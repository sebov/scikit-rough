import logging
from typing import Any, Callable, List, Literal, Optional, Sequence, TypeVar, overload

import pandas as pd

import skrough.typing as rght
from skrough.algorithms.exceptions import LoopBreak
from skrough.logs import log_start_end
from skrough.structs.state import GrowShrinkState

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=Callable)


@overload
def normalize_hook_sequence(
    hooks: rght.OneOrSequence[T],
    optional: Literal[False],
) -> Sequence[T]:
    ...


@overload
def normalize_hook_sequence(
    hooks: rght.OptionalOneOrSequence[T],
    optional: Literal[True],
) -> Optional[Sequence[T]]:
    ...


def normalize_hook_sequence(
    hooks: rght.OptionalOneOrSequence[T],
    optional: bool,
) -> Optional[Sequence[T]]:
    if optional:
        if (hooks is not None) and not isinstance(hooks, Sequence):
            hooks = [hooks]
    else:
        if hooks is None:
            raise ValueError("Hooks cannot be None")
        if not isinstance(hooks, Sequence):
            hooks = [hooks]
    return hooks


def aggregate_any_stop_hooks(
    hooks: rght.OneOrSequence[rght.StopHook],
):
    normalized_hooks = normalize_hook_sequence(hooks, optional=False)

    def _stop_function(
        state: GrowShrinkState,
        raise_exception: bool = False,
    ) -> bool:
        result = any(stop_hook(state) for stop_hook in normalized_hooks)
        if result and raise_exception:
            raise LoopBreak()
        return result

    return _stop_function


def aggregate_any_inner_stop_hooks(
    hooks: rght.OneOrSequence[rght.InnerStopHook],
):
    normalized_hooks = normalize_hook_sequence(hooks, optional=False)

    def _stop_function(
        state: GrowShrinkState,
        elements: rght.Elements,
        raise_exception: bool = False,
    ) -> bool:
        result = any(stop_hook(state, elements) for stop_hook in normalized_hooks)
        if result and raise_exception:
            raise LoopBreak()
        return result

    return _stop_function


# def aggregate_shrink_accept_hooks(
#     shrink_accept_group_index_hooks: Optional[
#         Sequence[rght.GSShrinkAcceptGroupIndexHook]
#     ],
# ):
#     def _shrink_accept_check(
#         state: GrowShrinkState,
#         group_index_to_check: GroupIndex,
#     ):
#         result = False
#         if shrink_accept_group_index_hooks is not None:
#             result = all(
#                 accept_hook(state, group_index_to_check)
#                 for accept_hook in shrink_accept_group_index_hooks
#             )
#         return result

#     return _shrink_accept_check


@log_start_end(logger)
def aggregate_update_state_hooks(
    hooks: rght.OptionalOneOrSequence[rght.UpdateStateHook],
):
    normalized_hooks = normalize_hook_sequence(hooks, optional=True)

    def _update_state_function(
        state: GrowShrinkState,
    ) -> None:
        if normalized_hooks is not None:
            for hook in normalized_hooks:
                hook(state)

    return _update_state_function


# @log_start_end(logger)
# def run_candidates_hooks(
#     state: GrowShrinkState,
#     elements: rght.Elements,
#     hooks: Optional[Sequence[rght.CandidatesHook]],
# ) -> rght.Elements:
#     if hooks is None:
#         logger.debug("No candidate hooks - using all elements")
#         candidates = elements
#     else:
#         logger.debug("Obtain candidate attrs using candidate attrs hooks")
#         candidates = np.fromiter(
#             itertools.chain.from_iterable(hook(state, elements) for hook in hooks),
#             dtype=np.int64,
#         )
#         # remove duplicates, preserve order of appearance
#         logger.debug("Remove duplicates from candidates")
#         candidates = np.unique(candidates)
#     logger.debug("Grow candidates count = %d", len(candidates))
#     return candidates


@log_start_end(logger)
def aggregate_produce_elements_hooks(
    hooks: rght.OptionalOneOrSequence[rght.ProduceElementsHook],
):
    normalized_hooks = normalize_hook_sequence(hooks, optional=True)

    def _produce_elements_function(
        state: GrowShrinkState,
    ) -> rght.Elements:
        result: List[Any] = []
        if normalized_hooks is not None:
            for hook in normalized_hooks:
                result.extend(hook(state))
        return pd.unique(result)

    return _produce_elements_function


@log_start_end(logger)
def aggregate_process_elements_hooks(
    hooks: rght.OptionalOneOrSequence[rght.ProcessElementsHook],
):
    normalized_hooks = normalize_hook_sequence(hooks, optional=True)

    def _process_elements_function(
        state: GrowShrinkState,
        elements: rght.Elements,
    ) -> rght.Elements:
        result: List[Any] = []
        if normalized_hooks is not None:
            for hook in normalized_hooks:
                result.extend(hook(state, elements))
        return pd.unique(result)

    return _process_elements_function


@log_start_end(logger)
def aggregate_chain_process_elements_hooks(
    hooks: rght.OptionalOneOrSequence[rght.ProcessElementsHook],
):
    normalized_hooks = normalize_hook_sequence(hooks, optional=True)

    def _process_elements_function(
        state: GrowShrinkState,
        elements: rght.Elements,
    ) -> rght.Elements:
        result = elements
        if normalized_hooks is not None:
            for hook in normalized_hooks:
                result = hook(state, result)
        return result

    return _process_elements_function
