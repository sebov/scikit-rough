"""Constants holding key names used in
:class:`~skrough.structs.state.ProcessingState` and its
:attr:`~skrough.structs.state.ProcessingState.input_data`,
:attr:`~skrough.structs.state.ProcessingState.config`,
:attr:`~skrough.structs.state.ProcessingState.values` attributes."""


CONFIG_KEYS_DOCSTRING_REGEX = "CONFIG_\\w*"
INPUT_DATA_KEYS_DOCSTRING_REGEX = "INPUT_DATA_\\w*"
VALUES_KEYS_DOCSTRING_REGEX = "VALUES_\\w*"


INPUT_DATA_X = "input_data_x"
"""A key used to reference the input data table representing conditional
features/attributes in ProcessingState.input_data."""

INPUT_DATA_X_COUNTS = "input_data_x_counts"
"""A key used to reference the input data table domain sizes in
ProcessingState.input_data."""

INPUT_DATA_Y = "input_data_y"
"""A key used to reference the input decision values in ProcessingState.input_data."""

INPUT_DATA_Y_COUNT = "input_data_y_count"
"""A key used to reference the input decision values domain size in
ProcessingState.input_data."""

CONFIG_CHAOS_FUN = "config_chaos_fun"
"""A key used to reference the chaos measure function to be used."""

CONFIG_CONSECUTIVE_EMPTY_ITERATIONS_MAX_COUNT = (
    "config_consecutive_empty_iterations_max_count"
)
CONFIG_DAAR_ALLOWED_RANDOMNESS = "config_daar_allowed_randomness"
CONFIG_DAAR_FAST = "config_daar_fast"
CONFIG_DAAR_PROBES_COUNT = "config_daar_probes_count"
CONFIG_DAAR_SMOOTHING_PARAMETER = "config_daar_smoothing"
CONFIG_EPSILON = "config_epsilon"
CONFIG_SELECT_ATTRS_CHAOS_SCORE_BASED_MAX_COUNT = (
    "config_select_attrs_chaos_score_based_max_count"
)
CONFIG_CANDIDATES_SELECT_RANDOM_MAX_COUNT = "config_candidates_select_random_max_count"
"""A key used to reference the max number of randomly selected candidate elements."""

CONFIG_RESULT_ATTRS_MAX_COUNT = "config_result_attrs_max_count"
"""A key used to reference the max number of attrs in the results."""

CONFIG_SET_APPROX_THRESHOLD_TO_CURRENT = "config_set_approx_threshold_to_current"
"""A key used to control the hook that sets approx threshold available under
ProcessingState.values['VALUES_CHAOS_SCORE_APPROX_THRESHOLD'] to the value resulting
from the current group index and decision values."""

VALUES_CHAOS_SCORE_APPROX_THRESHOLD = "values_chaos_score_approx_threshold"
"""A key used to reference the chaos score approximation threshold."""

VALUES_CHAOS_SCORE_BASE = "values_chaos_score_base"
"""A key used to reference the chaos score base value."""

VALUES_CHAOS_SCORE_TOTAL = "values_chaos_score_total"
"""A key used to reference the chaos score total value."""

VALUES_CONSECUTIVE_EMPTY_ITERATIONS_COUNT = "values_consecutive_empty_iterations_count"

VALUES_GROUP_INDEX = "values_group_index"
"""A key used to reference the current group index."""

VALUES_RESULT_ATTRS = "values_result_attrs"
"""A key used to reference the result attributes."""

VALUES_RESULT_OBJS = "values_result_objs"
"""A key used to reference the result objects."""

VALUES_X = "values_x"
"""A key used to reference the factorized data table representing conditional
features/attributes."""

VALUES_X_COUNTS = "values_x_counts"
"""A key used to reference the factorized data table domain sizes."""

VALUES_Y = "values_y"
"""A key used to reference the factorized decision values."""

VALUES_Y_COUNT = "values_y_count"
"""A key used to reference the factorized decision values domain size."""
