import logging

import skrough.typing as rght
from skrough.logs import log_start_end
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def inner_stop_hook_empty(
    state: ProcessingState,
    elements: rght.Elements,
) -> bool:
    return len(elements) == 0
