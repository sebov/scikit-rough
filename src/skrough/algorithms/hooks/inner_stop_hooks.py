import logging

import skrough.typing as rght
from skrough.algorithms.exceptions import LoopBreak
from skrough.logs import log_call
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_call
def inner_stop_hook_empty(
    state: ProcessingState,  # pylint: disable=unused-argument
    elements: rght.Elements,
) -> bool:
    return len(elements) == 0


@log_call
def inner_stop_hook_empty_loop_break(
    state: ProcessingState,  # pylint: disable=unused-argument
    elements: rght.Elements,
) -> bool:
    if len(elements) == 0:
        raise LoopBreak()
    return False
