import logging

import numpy as np

import skrough.typing as rght
from skrough.algorithms.exceptions import LoopBreak
from skrough.algorithms.hooks.names import HOOKS_DATA_X, HOOKS_RESULT_ATTRS
from skrough.logs import log_start_end
from skrough.structs.state import GrowShrinkState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def grow_pre_candidates_remaining_attrs(
    state: GrowShrinkState,
) -> rght.Elements:
    grow_pre_candidates: np.ndarray = np.delete(
        np.arange(state.values[HOOKS_DATA_X].shape[1]),
        state.values[HOOKS_RESULT_ATTRS],
    )
    if len(grow_pre_candidates) == 0:
        raise LoopBreak("No remaining attributes")
    return grow_pre_candidates
