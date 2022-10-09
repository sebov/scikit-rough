"""Constants holding key names used in
:class:`~skrough.structs.state.ProcessingState` and its
:attr:`~skrough.structs.state.ProcessingState.input_data`,
:attr:`~skrough.structs.state.ProcessingState.config`,
:attr:`~skrough.structs.state.ProcessingState.values` attributes."""


INPUT_X = "input_x"
"""A key used to reference the input data table representing conditional
features/attributes in ProcessingState.input_data."""

INPUT_Y = "input_y"
"""A key used to reference the decision values in ProcessingState.input_data."""

CONFIG_CANDIDATES_MAX_COUNT = "config_candidates_max_count"
CONFIG_CHAOS_FUN = "config_chaos_fun"
CONFIG_CONSECUTIVE_EMPTY_ITERATIONS_MAX_COUNT = (
    "config_consecutive_empty_iterations_max_count"
)
CONFIG_DAAR_ALLOWED_RANDOMNESS = "config_daar_allowed_randomness"
CONFIG_DAAR_N_OF_PROBES = "config_daar_n_of_probes"
CONFIG_EPSILON = "config_epsilon"
CONFIG_SELECT_ATTRS_CHAOS_SCORE_BASED_MAX_COUNT = (
    "config_select_attrs_chaos_score_based_max_count"
)
CONFIG_SELECT_RANDOM_MAX_COUNT = "config_select_attrs_random_max_count"
CONFIG_RESULT_ATTRS_MAX_COUNT = "config_result_attrs_max_count"


VALUES_CHAOS_SCORE_APPROX_THRESHOLD = "values_chaos_score_approx_value_threshold"
VALUES_CHAOS_SCORE_BASE = "values_chaos_score_base"
VALUES_CHAOS_SCORE_TOTAL = "values_chaos_score_total"
VALUES_CONSECUTIVE_EMPTY_ITERATIONS_COUNT = "values_consecutive_empty_iterations_count"
VALUES_GROUP_INDEX = "values_group_index"
VALUES_RESULT_ATTRS = "values_result_attrs"
VALUES_RESULT_OBJS = "values_result_objs"


VALUES_X = "values_x"
"""A key used to reference the factorized data table representing conditional
features/attributes."""

VALUES_X_COUNTS = "values_x_counts"
"""A key used to reference the factorized data table domain sizes."""

VALUES_Y = "values_y"
"""A key used to reference the factorized decision values."""

VALUES_Y_COUNT = "values_y_count"
"""A key used to reference the factorized decision values domain size."""
