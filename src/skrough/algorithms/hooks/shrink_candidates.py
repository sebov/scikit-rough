import logging

import numpy as np

from skrough.algorithms.hooks.names import RESULT_ATTRS
from skrough.logs import log_start_end
from skrough.structs.state import GrowShrinkState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def shrink_candidate_attrs_reversed(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
) -> np.ndarray:
    return np.asarray(list(reversed(state.values[RESULT_ATTRS])))
