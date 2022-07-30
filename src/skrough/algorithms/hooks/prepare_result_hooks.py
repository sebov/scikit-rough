import logging

from skrough.algorithms.hooks.names import HOOKS_RESULT_ATTRS, HOOKS_RESULT_OBJS
from skrough.logs import log_start_end
from skrough.structs.attrs_subset import AttrsSubset
from skrough.structs.objs_attrs_subset import ObjsAttrsSubset
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def prepare_result_hook_attrs_subset(
    state: ProcessingState,
) -> AttrsSubset:
    return AttrsSubset.create_from(state.values[HOOKS_RESULT_ATTRS])


@log_start_end(logger)
def prepare_result_hook_objs_attrs_subset(
    state: ProcessingState,
) -> ObjsAttrsSubset:
    return ObjsAttrsSubset.create_from(
        objs_like=state.values[HOOKS_RESULT_OBJS],
        attrs_like=state.values[HOOKS_RESULT_ATTRS],
    )
