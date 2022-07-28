import logging

from skrough.algorithms.hooks.names import (
    HOOKS_DATA_Y,
    HOOKS_DATA_Y_COUNT,
    HOOKS_GROUP_INDEX,
    HOOKS_RESULT_OBJS,
)
from skrough.instances import choose_objects
from skrough.logs import log_start_end
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def finalize_hook_choose_objs_random(
    state: ProcessingState,
) -> None:
    group_index = state.values[HOOKS_GROUP_INDEX]
    y = state.values[HOOKS_DATA_Y]
    y_count = state.values[HOOKS_DATA_Y_COUNT]
    result_objs = choose_objects(
        group_index=group_index,
        y=y,
        y_count=y_count,
        seed=state.rng,
    )
    logger.debug("Chosen objects count = %d", len(result_objs))
    state.values[HOOKS_RESULT_OBJS] = result_objs
