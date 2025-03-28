# pylint: disable=duplicate-code

from __future__ import annotations

import skrough.typing as rght
from skrough.algorithms import hooks
from skrough.algorithms.key_names import (
    CONFIG_CANDIDATES_SELECT_RANDOM_MAX_COUNT,
    CONFIG_CONSECUTIVE_EMPTY_ITERATIONS_MAX_COUNT,
    CONFIG_DAAR_ALLOWED_RANDOMNESS,
    CONFIG_DAAR_FAST,
    CONFIG_DAAR_PROBES_COUNT,
    CONFIG_DAAR_SMOOTHING_PARAMETER,
    CONFIG_DISORDER_FUN,
    CONFIG_EPSILON,
    CONFIG_RESULT_ATTRS_MAX_COUNT,
    CONFIG_SELECT_ATTRS_DISORDER_SCORE_BASED_MAX_COUNT,
    CONFIG_SET_APPROX_THRESHOLD_TO_CURRENT,
    INPUT_DATA_X,
    INPUT_DATA_X_COUNTS,
    INPUT_DATA_Y,
    INPUT_DATA_Y_COUNT,
)
from skrough.algorithms.meta import processing
from skrough.algorithms.reusables.attrs_daar import (
    attrs_daar_with_approx_and_count_stage,
)
from skrough.algorithms.reusables.attrs_greedy import attrs_greedy_stage
from skrough.algorithms.reusables.attrs_reduction import attrs_reduction_stage
from skrough.algorithms.reusables.objs_choose import objs_choose_randomly
from skrough.dataprep import prepare_factorized_array, prepare_factorized_vector

_get_bireduct_greedy_heuristic = processing.ProcessingMultiStage.from_hooks(
    init_multi_stage_hooks=[
        hooks.init_hooks.init_hook_pass_data,
        hooks.init_hooks.init_hook_single_group_index,
        hooks.init_hooks.init_hook_result_attrs_empty,
        hooks.init_hooks.init_hook_epsilon_approx_threshold,
    ],
    stages=[
        attrs_greedy_stage,
        objs_choose_randomly,
        attrs_reduction_stage,
    ],
    prepare_result_fun=hooks.prepare_result_hooks.prepare_result_hook_objs_attrs_subset,
)


def get_bireduct_greedy_heuristic(
    x,
    y,
    disorder_fun: rght.DisorderMeasure,
    epsilon: float,
    candidates_count: int | None = None,
    n_bireducts: int = 1,
    seed: rght.Seed = None,
    n_jobs: int | None = None,
):
    x, x_counts = prepare_factorized_array(x)
    y, y_count = prepare_factorized_vector(y)
    result = _get_bireduct_greedy_heuristic.call_parallel(
        n_times=n_bireducts,
        input_data={
            INPUT_DATA_X: x,
            INPUT_DATA_X_COUNTS: x_counts,
            INPUT_DATA_Y: y,
            INPUT_DATA_Y_COUNT: y_count,
        },
        config={
            CONFIG_DISORDER_FUN: disorder_fun,
            CONFIG_EPSILON: epsilon,
            CONFIG_SELECT_ATTRS_DISORDER_SCORE_BASED_MAX_COUNT: 1,
            CONFIG_CANDIDATES_SELECT_RANDOM_MAX_COUNT: candidates_count,
            CONFIG_SET_APPROX_THRESHOLD_TO_CURRENT: True,
        },
        seed=seed,
        n_jobs=n_jobs,
    )
    return result


_get_bireduct_daar_heuristic = processing.ProcessingMultiStage.from_hooks(
    init_multi_stage_hooks=[
        hooks.init_hooks.init_hook_pass_data,
        hooks.init_hooks.init_hook_single_group_index,
        hooks.init_hooks.init_hook_result_attrs_empty,
        hooks.init_hooks.init_hook_epsilon_approx_threshold,
    ],
    stages=[
        attrs_daar_with_approx_and_count_stage,
        objs_choose_randomly,
        attrs_reduction_stage,
    ],
    prepare_result_fun=hooks.prepare_result_hooks.prepare_result_hook_objs_attrs_subset,
)


def get_bireduct_daar_heuristic(
    x,
    y,
    disorder_fun: rght.DisorderMeasure,
    epsilon: float,
    attrs_max_count: int | None = None,
    candidates_count: int | None = None,
    selected_count: int | None = 1,
    consecutive_daar_reps: int = 1,
    allowed_randomness: float | None = None,
    probes_count: int | None = None,
    smoothing_parameter: float | None = None,
    fast: bool = False,
    n_bireducts: int = 1,
    seed: rght.Seed = None,
    n_jobs: int | None = None,
):
    x, x_counts = prepare_factorized_array(x)
    y, y_count = prepare_factorized_vector(y)

    n_attrs = x.shape[1]
    if allowed_randomness is None and n_attrs > 0:
        allowed_randomness = 1 / n_attrs
    if probes_count is None:
        probes_count = max(n_attrs, 100)

    input_data = {
        INPUT_DATA_X: x,
        INPUT_DATA_X_COUNTS: x_counts,
        INPUT_DATA_Y: y,
        INPUT_DATA_Y_COUNT: y_count,
    }

    config = {
        CONFIG_DISORDER_FUN: disorder_fun,
        CONFIG_EPSILON: epsilon,
        CONFIG_SELECT_ATTRS_DISORDER_SCORE_BASED_MAX_COUNT: selected_count,
        CONFIG_CANDIDATES_SELECT_RANDOM_MAX_COUNT: candidates_count,
        CONFIG_DAAR_ALLOWED_RANDOMNESS: allowed_randomness,
        CONFIG_DAAR_FAST: fast,
        CONFIG_DAAR_PROBES_COUNT: probes_count,
        CONFIG_CONSECUTIVE_EMPTY_ITERATIONS_MAX_COUNT: consecutive_daar_reps,
        CONFIG_SET_APPROX_THRESHOLD_TO_CURRENT: False,
        CONFIG_RESULT_ATTRS_MAX_COUNT: attrs_max_count,
    }
    if smoothing_parameter is not None:
        config[CONFIG_DAAR_SMOOTHING_PARAMETER] = smoothing_parameter

    result = _get_bireduct_daar_heuristic.call_parallel(
        n_times=n_bireducts,
        input_data=input_data,
        config=config,
        seed=seed,
        n_jobs=n_jobs,
    )
    return result
