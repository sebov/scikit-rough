import logging

import skrough.typing as rght
from skrough.logs import log_start_end
from skrough.structs.state import GrowShrinkState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def grow_inner_stop_empty(
    state: GrowShrinkState,
    elements: rght.Elements,
) -> bool:
    return len(elements) == 0
