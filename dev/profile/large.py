import cProfile
import pstats

import numpy as np
import pandas as pd

import skrough as rgh
from skrough.algorithms import hooks
from skrough.algorithms.key_names import (
    CONFIG_CANDIDATES_MAX_COUNT,
    CONFIG_CHAOS_FUN,
    CONFIG_CONSECUTIVE_EMPTY_ITERATIONS_MAX_COUNT,
    CONFIG_DAAR_ALLOWED_RANDOMNESS,
    CONFIG_DAAR_PROBES_COUNT,
    CONFIG_EPSILON,
    CONFIG_SELECT_ATTRS_CHAOS_SCORE_BASED_MAX_COUNT,
    INPUT_DATA_X,
    INPUT_DATA_Y,
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
from skrough.algorithms.meta import processing, stage
from skrough.chaos_measures import entropy, gini_impurity
from skrough.chaos_score import get_chaos_score_stats
from skrough.dataprep import prepare_factorized_array, prepare_factorized_vector
from skrough.structs.state import ProcessingState

df = pd.read_csv("../231_data_file.csv")
name = df.pop("name").to_numpy()
is_train = df.pop("is_train").to_numpy()
target = df.pop("target").to_numpy()
prediction = df.pop("prediction").to_numpy()
score_0 = df.pop("score_0").to_numpy()
score_1 = df.pop("score_1").to_numpy()

_, counts = prepare_factorized_array(df.to_numpy())
df = df.iloc[:, counts < 5]

df_train = df[is_train == 1]
target_train = target[is_train == 1]
df_test = df[is_train == 0]
target_test = target[is_train == 0]

x, x_counts = prepare_factorized_array(df_train.to_numpy())
y, y_count = prepare_factorized_vector(target_train)


grow_stage = stage.Stage.from_hooks(
    stop_hooks=[
        hooks.stop_hooks.stop_hook_approx_threshold,
        # hooks.stop_hooks.stop_hook_empty_iterations,
    ],
    init_hooks=None,
    pre_candidates_hooks=[
        hooks.pre_candidates_hooks.pre_candidates_hook_remaining_attrs,
    ],
    # candidates_hooks=hooks.common.process_elements.process_elements_hook_pass_everything,
    candidates_hooks=hooks.common.process_elements.create_process_elements_hook_random_choice(
        CONFIG_CANDIDATES_MAX_COUNT
    ),
    select_hooks=[
        hooks.select_hooks.select_hook_attrs_chaos_score_based,
    ],
    # filter_hooks=hooks.filter_hooks.filter_hook_attrs_first_daar,
    filter_hooks=None,
    # inner_init_hooks=hooks.inner_init_hooks.inner_init_hook_consecutive_empty_iterations_count,
    inner_init_hooks=None,
    inner_stop_hooks=hooks.inner_stop_hooks.inner_stop_hook_empty,
    inner_process_hooks=hooks.inner_process_hooks.inner_process_hook_add_first_attr,
    finalize_hooks=None,
)


msfun = processing.ProcessingMultiStage.from_hooks(
    init_multi_stage_hooks=[
        hooks.init_hooks.init_hook_factorize_data_x_y,
        hooks.init_hooks.init_hook_approx_threshold,
        hooks.init_hooks.init_hook_result_attrs_empty,
        hooks.init_hooks.init_hook_single_group_index,
    ],
    stages=[grow_stage],
    finalize_hooks=None,
    prepare_result_fun=hooks.prepare_result_hooks.prepare_result_hook_attrs_subset,
)


def compute():
    return msfun(
        input_data={
            INPUT_DATA_X: df_train.to_numpy(),
            INPUT_DATA_Y: target_train,
        },
        config={
            CONFIG_CHAOS_FUN: entropy,
            CONFIG_EPSILON: 0.15,
            CONFIG_SELECT_ATTRS_CHAOS_SCORE_BASED_MAX_COUNT: 1,
            CONFIG_CANDIDATES_MAX_COUNT: 10,
            # CONFIG_DAAR_ALLOWED_RANDOMNESS: 0.2,
            # CONFIG_DAAR_N_OF_PROBES: 100,
            # CONFIG_CONSECUTIVE_EMPTY_ITERATIONS_MAX_COUNT: 1,
        },
    )


compute()


def run():
    for i in range(10):
        print(i)
        compute()


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    run()
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats("./dev/profile/large.pstats")
