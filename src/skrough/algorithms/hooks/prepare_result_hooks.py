import logging

from skrough.algorithms.hooks.names import HOOKS_RESULT_ATTRS, HOOKS_RESULT_OBJS
from skrough.logs import log_start_end
from skrough.structs.objs_attrs_subset import AttrsSubset, ObjsAttrsSubset
from skrough.structs.state import GrowShrinkState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def prepare_result_hook_attrs_subset(
    state: GrowShrinkState,
) -> AttrsSubset:
    return AttrsSubset(attrs=state.values[HOOKS_RESULT_ATTRS])


@log_start_end(logger)
def prepare_result_hook_objs_attrs_subset(
    state: GrowShrinkState,
) -> ObjsAttrsSubset:
    return ObjsAttrsSubset(
        objs=state.values[HOOKS_RESULT_OBJS], attrs=state.values[HOOKS_RESULT_ATTRS]
    )
