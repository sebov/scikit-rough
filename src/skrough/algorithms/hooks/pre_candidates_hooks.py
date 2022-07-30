import logging

import numpy as np

import skrough.typing as rght
from skrough.algorithms.exceptions import LoopBreak
from skrough.algorithms.hooks.names import HOOKS_DATA_X, HOOKS_RESULT_ATTRS
from skrough.logs import log_start_end
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def pre_candidates_hook_remaining_attrs(
    state: ProcessingState,
) -> rght.Elements:
    pre_candidates: np.ndarray = np.delete(
        np.arange(state.values[HOOKS_DATA_X].shape[1]),
        state.values[HOOKS_RESULT_ATTRS],
    )
    if len(pre_candidates) == 0:
        raise LoopBreak("No remaining attrs")
    return pre_candidates
