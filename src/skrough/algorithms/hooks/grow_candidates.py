import logging

import numpy as np

from skrough.algorithms.hooks.names import GROW_CANDIDATE_ATTRS_MAX_COUNT
from skrough.logs import log_start_end
from skrough.structs.state import GrowShrinkState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def grow_candidate_attrs_random(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
    input_attrs: np.ndarray,
) -> np.ndarray:
    candidate_attrs_count = state.config.get(GROW_CANDIDATE_ATTRS_MAX_COUNT)
    if candidate_attrs_count is None:
        candidate_attrs = input_attrs.copy()
    else:
        candidate_attrs = state.rng.choice(
            input_attrs,
            min(len(input_attrs), candidate_attrs_count),
            replace=False,
        )
    logger.debug("candidate_attrs = %s", candidate_attrs)
    return candidate_attrs
