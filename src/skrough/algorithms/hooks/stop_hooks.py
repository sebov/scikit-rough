import logging

from skrough.algorithms.hooks.names import (
    HOOKS_CHAOS_FUN,
    HOOKS_CHAOS_SCORE_APPROX_THRESHOLD,
    HOOKS_CONSECUTIVE_EMPTY_ITERATIONS_COUNT,
    HOOKS_DATA_Y,
    HOOKS_DATA_Y_COUNT,
    HOOKS_EMPTY_ITERATIONS_MAX_COUNT,
    HOOKS_GROUP_INDEX,
    HOOKS_RESULT_ATTRS,
    HOOKS_RESULT_ATTRS_MAX_COUNT,
)
from skrough.logs import log_start_end
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def stop_hook_approx_threshold(
    state: ProcessingState,
) -> bool:
    """Stop check based on an expected level of approximation.

    Stop check based on an expected level of approximation. The function implements a
    check whether the current chaos score computed using the contents of the ``config``
    and ``values`` containers stored in the computation's state reached the expected
    approximation level maintained in the state object's values by some other
    complementary hook(s) under the ``HOOKS_APPROX_CHAOS_SCORE_VALUE_THRESHOLD`` key
    constant. The stop check uses the values stored under the following key constants to
    compute the current chaos score: ``HOOKS_CHAOS_FUN`` (``config``),
    ``HOOKS_GROUP_INDEX`` (``values``), ``HOOKS_DATA_Y`` (``values``),
    ``HOOKS_DATA_Y_COUNT`` (``values``).

    Args:
        state: State object that holds a computation's state.

    Returns:
        Indication whether the computation should stop.
    """
    group_index: GroupIndex = state.values[HOOKS_GROUP_INDEX]
    current_chaos_score = group_index.get_chaos_score(
        state.values[HOOKS_DATA_Y],
        state.values[HOOKS_DATA_Y_COUNT],
        state.config[HOOKS_CHAOS_FUN],
    )
    approx_chaos_score_value_threshold = state.values[
        HOOKS_CHAOS_SCORE_APPROX_THRESHOLD
    ]
    logger.debug("current_chaos_score = %f", current_chaos_score)
    logger.debug(
        "approx_chaos_score_value_threshold = %f", approx_chaos_score_value_threshold
    )
    return current_chaos_score <= approx_chaos_score_value_threshold


@log_start_end(logger)
def stop_hook_attrs_count(
    state: ProcessingState,
) -> bool:
    """Stop check based on a number of result attrs.

    Stop check based on a number of result attrs. The function implements a simple check
    whether the length of the result attrs collection (stored under the
    ``HOOKS_RESULT_ATTRS`` key constant) in the state object's values reached the limit
    setting (stored under the ``HOOKS_RESULT_ATTRS_MAX_COUNT`` key constant) from the
    state object's config.

    Args:
        state: State object that holds a computation's state.

    Returns:
        Indication whether the computation should stop.
    """
    return (
        len(state.values[HOOKS_RESULT_ATTRS])
        >= state.config[HOOKS_RESULT_ATTRS_MAX_COUNT]
    )


@log_start_end(logger)
def stop_hook_empty_iterations(
    state: ProcessingState,
) -> bool:
    """Stop check based on a number of empty iterations.

    Stop check based on a number of empty iterations. The function implements a simple
    check whether the number of iterations (ocurred in a row) which finished with the
    empty verified elements (most often as a result of
    ``pre_candidates-candidates-selected-verified`` function chain) reached the state
    object's config setting stored under the ``HOOKS_EMPTY_ITERATIONS_MAX_COUNT`` key
    constant. The number of the consecutive empty verified elements iterations is
    expected to be found under the ``HOOKS_EMPTY_ITERATIONS_COUNT`` key constant and is
    usually maintained in the state object's values by some other complementary hook(s).

    Args:
        state: State object that holds a computation's state.

    Returns:
        Indication whether the computation should stop.
    """
    return (
        state.values.get(HOOKS_CONSECUTIVE_EMPTY_ITERATIONS_COUNT, 0)
        >= state.config[HOOKS_EMPTY_ITERATIONS_MAX_COUNT]
    )
