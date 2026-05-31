import logging

from skrough.logs import log_call
from skrough.structs.attrs_subset import AttrsSubset
from skrough.structs.objs_attrs_subset import ObjsAttrsSubset
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_call
def prepare_result_hook_attrs_subset(
    state: ProcessingState,
) -> AttrsSubset:
    return AttrsSubset.from_attrs_like(state.get_values_result_attrs())


@log_call
def prepare_result_hook_objs_attrs_subset(
    state: ProcessingState,
) -> ObjsAttrsSubset:
    return ObjsAttrsSubset.from_objs_attrs_like(
        objs_like=state.get_values_result_objs(),
        attrs_like=state.get_values_result_attrs(),
    )
