import logging

import numpy as np

import skrough.typing as rght
from skrough.algorithms.exceptions import LoopBreak
from skrough.logs import log_start_end
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def pre_candidates_hook_remaining_attrs(
    state: ProcessingState,
) -> rght.Elements:
    pre_candidates: np.ndarray = np.delete(
        np.arange(state.get_values_x().shape[1]),
        state.get_values_result_attrs(),
    )
    if len(pre_candidates) == 0:
        raise LoopBreak("No remaining attrs")
    return pre_candidates


@log_start_end(logger)
def pre_candidates_hook_result_attrs(
    state: ProcessingState,
) -> rght.Elements:
    return state.get_values_result_attrs()
