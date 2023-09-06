import logging

from skrough.algorithms.hooks.helpers import check_if_below_approx_value_threshold
from skrough.algorithms.key_names import (
    CONFIG_CONSECUTIVE_EMPTY_ITERATIONS_MAX_COUNT,
    CONFIG_RESULT_ATTRS_MAX_COUNT,
    VALUES_CONSECUTIVE_EMPTY_ITERATIONS_COUNT,
    VALUES_GROUP_INDEX,
    VALUES_RESULT_ATTRS,
)
from skrough.logs import log_start_end
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def dummy_stop_hook(
    state: ProcessingState,
) -> bool:
    """Dummy stop hook function.

    The function raises not implemented error exception. It is a special hook function
    that may be used in situations when one wants to define a base/template Stage that
    will be cloned and adjusted to form actual stages that differ very little from the
    base/template one. This is needed because a Stage instance needs at least one stop
    hook function to be given.
    """
    raise NotImplementedError


@log_start_end(logger)
def stop_hook_approx_threshold(
    state: ProcessingState,
) -> bool:
    """Check if the defined chaos score approximation threshold was reached.

    The function checks if the defined level of chaos score approximation is reached.
    The function is intended for use in cases of chaos score minimizing processes and
    therefore it check if the chaos score value computed (cf.
    :func:`~skrough.structs.group_index.GroupIndex.get_chaos_score` and
    :mod:`~skrough.chaos_score` module) for the current group index falls below the
    defined level of chaos score approximation.

    The function uses the following config and intermediate mappings stored in the
    ``state`` argument and appropriate keys to access the actual values:

    - config values (:attr:`skrough.structs.state.ProcessingState.config` mapping):

        - chaos measure function (cf. :mod:`~skrough.chaos_measures.chaos_measures`) to
          be used in chaos score computation - accessed using
          :const:`~skrough.algorithms.key_names.CONFIG_CHAOS_FUN` key

    - intermediate values (:attr:`skrough.structs.state.ProcessingState.values`
      mapping)

        - chaos score approximation threshold - accessed using
          :const:`~skrough.algorithms.key_names.VALUES_CHAOS_SCORE_APPROX_THRESHOLD`
          key
        - group index to be used in chaos score computation - accessed using
          :const:`~skrough.algorithms.key_names.VALUES_GROUP_INDEX` key
        - factorized values of the target attribute - accessed using
          :const:`~skrough.algorithms.key_names.VALUES_Y` key
        - number of distinct values of the target attribute - accessed using
          :const:`~skrough.algorithms.key_names.VALUES_Y_COUNT` key

    Args:
        state: State object that holds the computation's state.

    Returns:
        Indication whether the chaos score computed for the current group index falls
        below the defined chaos score approximation threshold.
    """
    group_index: GroupIndex = state.values[VALUES_GROUP_INDEX]
    return check_if_below_approx_value_threshold(state, group_index)


# TODO: add description for max_count == None ~ no limit
@log_start_end(logger)
def stop_hook_attrs_count(
    state: ProcessingState,
) -> bool:
    """Stop check based on a number of result attrs.

    Stop check based on a number of result attrs. The function implements a simple check
    whether the length of the result attrs collection (stored under the
    :const:`~skrough.algorithms.key_names.VALUES_RESULT_ATTRS` key constant) in the
    state object's values reached the limit setting (stored under the
    :const:`~skrough.algorithms.key_names.CONFIG_RESULT_ATTRS_MAX_COUNT` key constant)
    from the state object's config.

    Args:
        state: State object that holds a computation's state.

    Returns:
        Indication whether the computation should stop.
    """
    result = False
    attrs_max_count = state.config.get(CONFIG_RESULT_ATTRS_MAX_COUNT)
    if attrs_max_count is not None:
        result = bool(
            len(state.values[VALUES_RESULT_ATTRS])
            >= state.config[CONFIG_RESULT_ATTRS_MAX_COUNT]
        )
    return result


@log_start_end(logger)
def stop_hook_empty_iterations(
    state: ProcessingState,
) -> bool:
    """Stop check based on a number of empty iterations.

    Stop check based on a number of empty iterations. The function implements a simple
    check whether the number of iterations (ocurred in a row) which finished with the
    empty verified elements (most often as a result of
    ``pre_candidates-candidates-selected-verified`` function chain) reached the state
    object's config setting stored under the
    :const:`~skrough.algorithms.key_names.CONFIG_CONSECUTIVE_EMPTY_ITERATIONS_MAX_COUNT`
    key constant. The number of the consecutive empty verified elements iterations is
    expected to be found under the
    :const:`~skrough.algorithms.key_names.VALUES_CONSECUTIVE_EMPTY_ITERATIONS_COUNT` key
    constant and is usually maintained in the state object's values by some other
    complementary hook(s).

    Args:
        state: State object that holds a computation's state.

    Returns:
        Indication whether the computation should stop.
    """
    return bool(
        state.values.get(VALUES_CONSECUTIVE_EMPTY_ITERATIONS_COUNT, 0)
        >= state.config[CONFIG_CONSECUTIVE_EMPTY_ITERATIONS_MAX_COUNT]
    )


@log_start_end(logger)
def stop_hook_always_false(
    state: ProcessingState,  # pylint: disable=unused-argument
) -> bool:
    return False


@log_start_end(logger)
def stop_hook_always_true(
    state: ProcessingState,  # pylint: disable=unused-argument
) -> bool:
    return True
