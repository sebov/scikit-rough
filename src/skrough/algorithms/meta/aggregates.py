import logging
from typing import Sequence

from attrs import define

import skrough.typing as rght
from skrough.algorithms.exceptions import LoopBreak
from skrough.algorithms.meta.helpers import normalize_hook_sequence
from skrough.logs import log_start_end
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@define
class StopHooksAggregate:
    normalized_hooks: Sequence[rght.StopHook]

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
class InnerStopHooksAggregate:
    normalized_hooks: Sequence[rght.InnerStopHook]

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
