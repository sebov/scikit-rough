import logging
from dataclasses import dataclass

from skrough.instances import choose_objects
from skrough.logs import log_start_end
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@dataclass
class FinalizeHookChooseObjsRandomly:
    @log_start_end(logger)
    def __call__(
        self,
        state: ProcessingState,
    ) -> None:
        group_index = state.get_values_group_index()
        y = state.get_values_y()
        y_count = state.get_values_y_count()
        result_objs = choose_objects(
            group_index=group_index,
            y=y,
            y_count=y_count,
            seed=state.get_rng(),
        )
        logger.debug("Chosen objects count = %d", len(result_objs))
        state.set_values_result_objs(result_objs)
