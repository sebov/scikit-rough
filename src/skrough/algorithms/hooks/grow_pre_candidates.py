import logging

import numpy as np

from skrough.algorithms.exceptions import LoopBreak
from skrough.algorithms.hooks.names import DATA_X, RESULT_ATTRS
from skrough.logs import log_start_end
from skrough.structs.state import GrowShrinkState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def grow_pre_candidates_remaining_attrs(
    state: GrowShrinkState,
) -> np.ndarray:
    grow_pre_candidates: np.ndarray = np.delete(
        np.arange(state.values[DATA_X].shape[1]),
        state.values[RESULT_ATTRS],
    )
    if len(grow_pre_candidates) == 0:
        raise LoopBreak("No remaining attributes")
    return grow_pre_candidates
