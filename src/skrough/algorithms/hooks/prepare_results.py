import logging

import numpy as np

from skrough.algorithms.hooks.names import RESULT_ATTRS, RESULT_OBJS
from skrough.logs import log_start_end
from skrough.structs.objs_attrs_subset import AttrsSubset, ObjsAttrsSubset
from skrough.structs.state import GrowShrinkState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def prepare_result_attrs_subset(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
) -> AttrsSubset:
    return AttrsSubset(attrs=state.values[RESULT_ATTRS])


@log_start_end(logger)
def prepare_result_objs_attrs_subset(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
) -> ObjsAttrsSubset:
    return ObjsAttrsSubset(
        objs=state.values[RESULT_OBJS], attrs=state.values[RESULT_ATTRS]
    )
