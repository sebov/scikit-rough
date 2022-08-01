import logging

import skrough.typing as rght
from skrough.algorithms.hooks.names import CONFIG_CANDIDATES_MAX_COUNT
from skrough.logs import log_start_end
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def candidates_hook_random_choice(
    state: ProcessingState,
    elements: rght.Elements,
) -> rght.Elements:
    candidates_count = state.config.get(CONFIG_CANDIDATES_MAX_COUNT)
    if candidates_count is None:
        candidates_count = len(elements)
    candidates_attrs_count = min(len(elements), candidates_count)
    candidates = state.rng.choice(
        elements,
        size=candidates_attrs_count,
        replace=False,
    )
    logger.debug("candidates = %s", candidates)
    return candidates
