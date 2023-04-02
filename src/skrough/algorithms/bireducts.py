# pylint: disable=duplicate-code

from typing import Optional

import skrough.typing as rght
from skrough.algorithms import hooks
from skrough.algorithms.key_names import (
    CONFIG_CANDIDATES_SELECT_RANDOM_MAX_COUNT,
    CONFIG_CHAOS_FUN,
    CONFIG_EPSILON,
    CONFIG_SELECT_ATTRS_CHAOS_SCORE_BASED_MAX_COUNT,
    INPUT_DATA_X,
    INPUT_DATA_Y,
)
from skrough.algorithms.meta import processing
from skrough.algorithms.reusables.attrs_greedy import greedy_stage
from skrough.algorithms.reusables.attrs_reduction import reduction_stage

_get_bireduct_greedy_heuristic = processing.ProcessingMultiStage.from_hooks(
    init_multi_stage_hooks=[
        hooks.init_hooks.init_hook_factorize_data_x_y,
        hooks.init_hooks.init_hook_single_group_index,
        hooks.init_hooks.init_hook_result_attrs_empty,
        hooks.init_hooks.init_hook_approx_threshold,
    ],
    stages=[greedy_stage, reduction_stage],
    finalize_hooks=hooks.finalize_hooks.finalize_hook_choose_objs_randomly,
    prepare_result_fun=hooks.prepare_result_hooks.prepare_result_hook_objs_attrs_subset,
)


def get_bireduct_greedy_heuristic(
    x,
    y,
    epsilon: float,
    candidates_count: Optional[int],
    chaos_measure: rght.ChaosMeasure,
    seed: rght.Seed = None,
):
    return _get_bireduct_greedy_heuristic(
        input_data={
            INPUT_DATA_X: x,
            INPUT_DATA_Y: y,
        },
        config={
            CONFIG_CHAOS_FUN: chaos_measure,
            CONFIG_EPSILON: epsilon,
            CONFIG_SELECT_ATTRS_CHAOS_SCORE_BASED_MAX_COUNT: 1,
            CONFIG_CANDIDATES_SELECT_RANDOM_MAX_COUNT: candidates_count,
        },
        seed=seed,
    )
