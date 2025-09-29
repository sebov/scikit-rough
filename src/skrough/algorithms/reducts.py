# pylint: disable=duplicate-code

from __future__ import annotations

import skrough.typing as rght
from skrough.algorithms import hooks
from skrough.algorithms.meta import processing
from skrough.algorithms.reusables.attrs_daar import attrs_daar_stage
from skrough.algorithms.reusables.attrs_greedy import attrs_greedy_stage
from skrough.algorithms.reusables.attrs_reduction import attrs_reduction_stage
from skrough.dataprep import prepare_factorized_array, prepare_factorized_vector
from skrough.structs.state import ProcessingState

_get_approx_reduct_greedy_heuristic = processing.ProcessingMultiStage.from_hooks(
    init_multi_stage_hooks=[
        hooks.init_hooks.init_hook_pass_data,
        hooks.init_hooks.init_hook_single_group_index,
        hooks.init_hooks.init_hook_result_attrs_empty,
        hooks.init_hooks.init_hook_epsilon_approx_threshold,
    ],
    stages=[attrs_greedy_stage, attrs_reduction_stage],
    finalize_hooks=None,
    prepare_result_fun=hooks.prepare_result_hooks.prepare_result_hook_attrs_subset,
)


def get_approx_reduct_greedy_heuristic(
    x,
    y,
    disorder_fun: rght.DisorderMeasure,
    epsilon: float,
    candidates_count: int | None = None,
    n_reducts: int = 1,
    seed: rght.Seed = None,
    n_jobs: int | None = None,
):
    x, x_counts = prepare_factorized_array(x)
    y, y_count = prepare_factorized_vector(y)

    state = ProcessingState.from_optional(
        processing_fun=None,
        rng=None,
    )
    state.set_input_data_x(x)
    state.set_input_data_x_counts(x_counts)
    state.set_input_data_y(y)
    state.set_input_data_y_count(y_count)
    state.set_config_disorder_fun(disorder_fun)
    state.set_config_epsilon(epsilon)
    state.set_config_select_attrs_disorder_score_based_max_count(1)
    if candidates_count is not None:
        state.set_config_candidates_select_random_max_count(candidates_count)

    result = _get_approx_reduct_greedy_heuristic.call_parallel(
        n_times=n_reducts,
        state=state,
        seed=seed,
        n_jobs=n_jobs,
    )
    return result


_get_approx_reduct_daar_heuristic = processing.ProcessingMultiStage.from_hooks(
    init_multi_stage_hooks=[
        hooks.init_hooks.init_hook_pass_data,
        hooks.init_hooks.init_hook_single_group_index,
        hooks.init_hooks.init_hook_result_attrs_empty,
    ],
    stages=[attrs_daar_stage, attrs_reduction_stage],
    finalize_hooks=None,
    prepare_result_fun=hooks.prepare_result_hooks.prepare_result_hook_attrs_subset,
)


def get_approx_reduct_daar_heuristic(
    x,
    y,
    disorder_fun: rght.DisorderMeasure,
    candidates_count: int | None = None,
    consecutive_daar_reps: int = 1,
    allowed_randomness: float | None = None,
    probes_count: int | None = None,
    smoothing_parameter: float | None = None,
    fast: bool = False,
    n_reducts: int = 1,
    seed: rght.Seed = None,
    n_jobs: int | None = None,
):
    x, x_counts = prepare_factorized_array(x)
    y, y_count = prepare_factorized_vector(y)

    n_attrs = max(1, x.shape[1])
    if allowed_randomness is None:
        allowed_randomness = 1 / n_attrs
    if probes_count is None:
        probes_count = max(n_attrs, 100)

    state = ProcessingState.from_optional(
        processing_fun=None,
        rng=None,
    )
    state.set_input_data_x(x)
    state.set_input_data_x_counts(x_counts)
    state.set_input_data_y(y)
    state.set_input_data_y_count(y_count)
    state.set_config_disorder_fun(disorder_fun)
    state.set_config_select_attrs_disorder_score_based_max_count(1)
    if candidates_count is not None:
        state.set_config_candidates_select_random_max_count(candidates_count)
    state.set_config_daar_allowed_randomness(allowed_randomness)
    state.set_config_daar_fast(fast)
    state.set_config_daar_probes_count(probes_count)
    state.set_config_consecutive_empty_iterations_max_count(consecutive_daar_reps)
    state.set_config_set_approx_threshold_to_current(True)
    if smoothing_parameter is not None:
        state.set_config_daar_smoothing_parameter(smoothing_parameter)

    result = _get_approx_reduct_daar_heuristic.call_parallel(
        n_times=n_reducts,
        state=state,
        seed=seed,
        n_jobs=n_jobs,
    )
    return result
