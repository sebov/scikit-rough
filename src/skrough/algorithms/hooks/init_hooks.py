"""Init hook functions."""

import logging

from skrough.algorithms.key_names import (
    CONFIG_CHAOS_FUN,
    CONFIG_EPSILON,
    CONFIG_SET_APPROX_THRESHOLD_TO_CURRENT,
    INPUT_DATA_X,
    INPUT_DATA_X_COUNTS,
    INPUT_DATA_Y,
    INPUT_DATA_Y_COUNT,
    VALUES_CHAOS_SCORE_APPROX_THRESHOLD,
    VALUES_CHAOS_SCORE_BASE,
    VALUES_CHAOS_SCORE_TOTAL,
    VALUES_GROUP_INDEX,
    VALUES_RESULT_ATTRS,
    VALUES_RESULT_OBJS,
    VALUES_X,
    VALUES_X_COUNTS,
    VALUES_Y,
    VALUES_Y_COUNT,
)
from skrough.chaos_score import get_chaos_score_stats
from skrough.dataprep import prepare_factorized_array, prepare_factorized_vector
from skrough.logs import log_start_end
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


@log_start_end(logger)
def init_hook_factorize_data_x_y(
    state: ProcessingState,
) -> None:
    """Init hook function to factorize the input data.

    Factorize an input data table representing conditional features/attributes and
    decision values for the latter computations. It is assumed that that the input data
    array and decision values are available in :attr:`state.input_data` under
    :const:`~skrough.algorithms.key_names.INPUT_DATA_X` and
    :const:`~skrough.algorithms.key_names.INPUT_DATA_Y` keys, respectively.

    The :func:`skrough.dataprep.prepare_factorized_array` function is used to process
    the input data table and the corresponding results are stored in
    :attr:`state.values` under :const:`~skrough.algorithms.key_names.VALUES_X` and
    :const:`~skrough.algorithms.key_names.VALUES_X_COUNTS` keys.

    The :func:`skrough.dataprep.prepare_factorized_vector` function is used to process
    the decision values and the corresponding results are stored in :attr:`state.values`
    under :const:`~skrough.algorithms.key_names.VALUES_Y` and
    :const:`~skrough.algorithms.key_names.VALUES_Y_COUNT` keys.

    Args:
        state: An object representing the processing state.
    """
    x, x_counts = prepare_factorized_array(state.input_data[INPUT_DATA_X])
    y, y_count = prepare_factorized_vector(state.input_data[INPUT_DATA_Y])
    state.values[VALUES_X] = x
    state.values[VALUES_X_COUNTS] = x_counts
    state.values[VALUES_Y] = y
    state.values[VALUES_Y_COUNT] = y_count


# TODO: add docstring
@log_start_end(logger)
def init_hook_pass_data(
    state: ProcessingState,
) -> None:
    state.values[VALUES_X] = state.input_data[INPUT_DATA_X]
    state.values[VALUES_X_COUNTS] = state.input_data[INPUT_DATA_X_COUNTS]
    state.values[VALUES_Y] = state.input_data[INPUT_DATA_Y]
    state.values[VALUES_Y_COUNT] = state.input_data[INPUT_DATA_Y_COUNT]


# TODO: update docstring
@log_start_end(logger)
def init_hook_single_group_index(
    state: ProcessingState,
) -> None:
    """Init hook function to initialize a uniform group index structure.

    It is assumed that the appropriate data set that is consisted of objects (typically
    rows of some tabular representation) is available in :attr:`state.values` under the
    :const:`~skrough.algorithms.key_names.VALUES_X` key. The function initializes a
    uniform group index, i.e., a group index that assigns each of the objects under
    consideration to the same group.

    The group index will be stored in :attr:`state.values` under the
    :const:`~skrough.algorithms.key_names.VALUES_GROUP_INDEX` key.

    Args:
        state: An object representing the processing state.
    """
    group_index = GroupIndex.create_uniform(len(state.values[VALUES_X]))
    state.values[VALUES_GROUP_INDEX] = group_index


@log_start_end(logger)
def init_hook_result_objs_empty(
    state: ProcessingState,
) -> None:
    """Init hook function to initialize an empty objects locations collection.

    The function initializes an empty objects locations list and stores it in
    :attr:`state.values` under the
    :const:`~skrough.algorithms.key_names.VALUES_RESULT_OBJS` key.

    The initialized list is intended to be used as integer-location based indexing
    sequence of objects, i.e., 0-based values that index objects from the considered
    data set.

    Args:
        state: An object representing the processing state.
    """
    state.values[VALUES_RESULT_OBJS] = []


@log_start_end(logger)
def init_hook_result_attrs_empty(
    state: ProcessingState,
) -> None:
    """Init hook function to initialize an empty attributes locations collection.

    The function initializes an empty attributes locations list and stores it in
    :attr:`state.values` under the
    :const:`~skrough.algorithms.key_names.VALUES_RESULT_ATTRS` key.

    The initialized list is intended to be used as integer-location based indexing
    sequence of attributes, i.e., 0-based values that index attributes from the
    considered data set.

    Args:
        state: An object representing the processing state.
    """
    state.values[VALUES_RESULT_ATTRS] = []


@log_start_end(logger)
def init_hook_epsilon_approx_threshold(
    state: ProcessingState,
) -> None:
    chaos_stats = get_chaos_score_stats(
        x=state.values[VALUES_X],
        x_counts=state.values[VALUES_X_COUNTS],
        y=state.values[VALUES_Y],
        y_count=state.values[VALUES_Y_COUNT],
        chaos_fun=state.config[CONFIG_CHAOS_FUN],
        epsilon=state.config[CONFIG_EPSILON],
    )
    state.values.update(
        {
            VALUES_CHAOS_SCORE_BASE: chaos_stats.base,
            VALUES_CHAOS_SCORE_TOTAL: chaos_stats.total,
            VALUES_CHAOS_SCORE_APPROX_THRESHOLD: chaos_stats.approx_threshold,
        }
    )


@log_start_end(logger)
def init_hook_current_approx_threshold(
    state: ProcessingState,
) -> None:
    if state.config.get(CONFIG_SET_APPROX_THRESHOLD_TO_CURRENT) is True:
        group_index: GroupIndex = state.values[VALUES_GROUP_INDEX]
        approx_threshold = group_index.get_chaos_score(
            values=state.values[VALUES_Y],
            values_count=state.values[VALUES_Y_COUNT],
            chaos_fun=state.config[CONFIG_CHAOS_FUN],
        )
        state.values.update({VALUES_CHAOS_SCORE_APPROX_THRESHOLD: approx_threshold})
