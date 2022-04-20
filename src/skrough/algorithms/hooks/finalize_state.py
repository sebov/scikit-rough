import logging

import numpy as np

from skrough.algorithms.hooks.names import RESULT_OBJS, SINGLE_GROUP_INDEX
from skrough.instances import choose_objects
from skrough.logs import log_start_end
from skrough.structs.state import GrowShrinkState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def finalize_state_draw_objects(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
) -> None:
    result_objs = choose_objects(
        state.values[SINGLE_GROUP_INDEX],
        y,
        y_count,
        seed=state.rng,
    )
    logger.debug("Chosen objects count = %d", len(result_objs))
    state.values[RESULT_OBJS] = result_objs
