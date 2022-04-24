import logging

import skrough.typing as rght
from skrough.algorithms.hooks.names import GROW_CANDIDATES_MAX_COUNT
from skrough.logs import log_start_end
from skrough.structs.state import GrowShrinkState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def candidates_hooks_grow_attrs_random(
    state: GrowShrinkState,
    elements: rght.Elements,
) -> rght.Elements:
    candidate_attrs_count = state.config.get(GROW_CANDIDATES_MAX_COUNT)
    if candidate_attrs_count is None:
        candidates = elements
    else:
        candidates = state.rng.choice(
            elements,
            min(len(elements), candidate_attrs_count),
            replace=False,
        )
    logger.debug("candidates = %s", candidates)
    return candidates
