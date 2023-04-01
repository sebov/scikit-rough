import logging

from skrough.algorithms.key_names import (
    VALUES_GROUP_INDEX,
    VALUES_RESULT_OBJS,
    VALUES_Y,
    VALUES_Y_COUNT,
)
from skrough.instances import choose_objects
from skrough.logs import log_start_end
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def finalize_hook_choose_objs_randomly(
    state: ProcessingState,
) -> None:
    group_index = state.values[VALUES_GROUP_INDEX]
    y = state.values[VALUES_Y]
    y_count = state.values[VALUES_Y_COUNT]
    result_objs = choose_objects(
        group_index=group_index,
        y=y,
        y_count=y_count,
        seed=state.rng,
    )
    logger.debug("Chosen objects count = %d", len(result_objs))
    state.values[VALUES_RESULT_OBJS] = result_objs
