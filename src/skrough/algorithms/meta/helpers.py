import itertools
import logging
from typing import Callable, Literal, Optional, Sequence, TypeVar, Union, overload

import numpy as np

import skrough.typing as rght
from skrough.algorithms.exceptions import LoopBreak
from skrough.logs import log_start_end
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import GrowShrinkState

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=Callable)


@overload
def normalize_hook_sequence(
    hooks: Union[T, Sequence[T]],
    optional: Literal[False],
) -> Sequence[T]:
    ...


@overload
def normalize_hook_sequence(
    hooks: Optional[Union[T, Sequence[T]]],
    optional: Literal[True],
) -> Optional[Sequence[T]]:
    ...


def normalize_hook_sequence(
    hooks: Optional[Union[T, Sequence[T]]],
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


def aggregate_grow_stop_hooks(
    grow_stop_hooks: Sequence[rght.GSGrowStopHook],
):
    def _grow_stop_check(
        state: GrowShrinkState,
    ):
        if any(stop_hook(state) for stop_hook in grow_stop_hooks):
            raise LoopBreak()

    return _grow_stop_check


def aggregate_shrink_accept_hooks(
    shrink_accept_group_index_hooks: Optional[
        Sequence[rght.GSShrinkAcceptGroupIndexHook]
    ],
):
    def _shrink_accept_check(
        state: GrowShrinkState,
        group_index_to_check: GroupIndex,
    ):
        result = False
        if shrink_accept_group_index_hooks is not None:
            result = all(
                accept_hook(state, group_index_to_check)
                for accept_hook in shrink_accept_group_index_hooks
            )
        return result

    return _shrink_accept_check


@log_start_end(logger)
def run_update_hooks(
    state: GrowShrinkState, hooks: Optional[Sequence[rght.GSUpdateStateHook]]
):
    if hooks is not None:
        for hook in hooks:
            hook(state)


@log_start_end(logger)
def run_candidates_hooks(
    state: GrowShrinkState,
    elements: rght.GSElements,
    hooks: Optional[Sequence[rght.GSGrowCandidatesHook]],
) -> rght.GSElements:
    if hooks is None:
        logger.debug("No candidate hooks - using all elements")
        candidates = elements
    else:
        logger.debug("Obtain candidate attrs using candidate attrs hooks")
        candidates = np.fromiter(
            itertools.chain.from_iterable(hook(state, elements) for hook in hooks),
            dtype=np.int64,
        )
        # remove duplicates, preserve order of appearance
        logger.debug("Remove duplicates from candidates")
        candidates = np.unique(candidates)
    logger.debug("Grow candidates count = %d", len(candidates))
    return candidates
