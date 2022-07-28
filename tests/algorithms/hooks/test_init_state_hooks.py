import numpy as np
import pandas as pd
import pytest

from skrough.algorithms.hooks.init_state_hooks import (
    init_state_hook_approx_threshold,
    init_state_hook_factorize_data_x_y,
    init_state_hook_result_attrs_empty,
    init_state_hook_result_objs_empty,
    init_state_hook_single_group_index,
)
from skrough.algorithms.hooks.names import (
    HOOKS_CHAOS_FUN,
    HOOKS_CHAOS_SCORE_APPROX_THRESHOLD,
    HOOKS_CHAOS_SCORE_BASE,
    HOOKS_CHAOS_SCORE_TOTAL,
    HOOKS_DATA_X,
    HOOKS_DATA_X_COUNTS,
    HOOKS_DATA_Y,
    HOOKS_DATA_Y_COUNT,
    HOOKS_EPSILON,
    HOOKS_GROUP_INDEX,
    HOOKS_INPUT_X,
    HOOKS_INPUT_Y,
    HOOKS_RESULT_ATTRS,
    HOOKS_RESULT_OBJS,
)
from skrough.chaos_measures import conflicts_number, entropy, gini_impurity
from skrough.chaos_score import get_chaos_score_for_data
from skrough.dataprep import (
    prepare_factorized_array,
    prepare_factorized_data,
    prepare_factorized_vector,
)
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import ProcessingState


@pytest.mark.parametrize(
    "data",
    [
        [
            [0, 1],
            [1, 0],
        ],
        np.empty(shape=(0, 1)),
        np.empty(shape=(1, 1)),
        np.empty(shape=(5, 1)),
        np.empty(shape=(0, 2)),
        np.empty(shape=(1, 2)),
        np.empty(shape=(5, 2)),
        np.empty(shape=(0, 5)),
        np.empty(shape=(1, 5)),
        np.empty(shape=(5, 5)),
    ],
)
def test_state_hook_factorize_data_x_y(data, state_fixture: ProcessingState):
    df = pd.DataFrame(data)
    x, x_counts, y, y_count = prepare_factorized_data(df, df.shape[1] - 1)

    state_fixture.input_data = {
        HOOKS_INPUT_X: df.iloc[:, :-1].to_numpy(),
        HOOKS_INPUT_Y: df.iloc[:, -1].to_numpy(),
    }
    init_state_hook_factorize_data_x_y(state_fixture)

    assert state_fixture.values.keys() == {
        HOOKS_DATA_X,
        HOOKS_DATA_X_COUNTS,
        HOOKS_DATA_Y,
        HOOKS_DATA_Y_COUNT,
    }
    assert np.array_equal(state_fixture.values[HOOKS_DATA_X], x)
    assert np.array_equal(state_fixture.values[HOOKS_DATA_X_COUNTS], x_counts)
    assert np.array_equal(state_fixture.values[HOOKS_DATA_Y], y)
    assert np.array_equal(state_fixture.values[HOOKS_DATA_Y_COUNT], y_count)


@pytest.mark.parametrize(
    "data",
    [
        np.empty(shape=(0, 0)),
        np.empty(shape=(0, 1)),
        np.empty(shape=(1, 0)),
        np.empty(shape=(2, 2)),
        np.empty(shape=(4, 1)),
        np.empty(shape=(4, 3)),
    ],
)
def test_init_state_hook_single_group_index(data, state_fixture: ProcessingState):
    state_fixture.values = {HOOKS_DATA_X: data}
    assert HOOKS_GROUP_INDEX not in state_fixture.values
    init_state_hook_single_group_index(state_fixture)
    assert HOOKS_GROUP_INDEX in state_fixture.values
    group_index: GroupIndex = state_fixture.values[HOOKS_GROUP_INDEX]
    assert group_index.n_objs == len(data)
    assert group_index.n_groups == (1 if len(data) > 0 else 0)


def test_init_state_hook_result_objs_empty(state_fixture: ProcessingState):
    state_fixture.values = {}
    init_state_hook_result_objs_empty(state_fixture)
    assert state_fixture.values == {HOOKS_RESULT_OBJS: []}
    # let's invoke for the second time
    init_state_hook_result_objs_empty(state_fixture)
    assert state_fixture.values == {HOOKS_RESULT_OBJS: []}


def test_init_state_hook_result_attrs_empty(state_fixture: ProcessingState):
    state_fixture.values = {}
    init_state_hook_result_attrs_empty(state_fixture)
    assert state_fixture.values == {HOOKS_RESULT_ATTRS: []}
    # let's invoke for the second time
    init_state_hook_result_attrs_empty(state_fixture)
    assert state_fixture.values == {HOOKS_RESULT_ATTRS: []}


@pytest.mark.parametrize("chaos_fun", [conflicts_number, entropy, gini_impurity])
@pytest.mark.parametrize("epsilon", [0.0, 0.1, 0.9, 1.0])
@pytest.mark.parametrize(
    "x, y",
    [
        ([[0], [1]], [0, 1]),
        (np.empty(shape=(0, 0)), np.empty(0)),
        (np.empty(shape=(0, 4)), np.empty(0)),
        (np.empty(shape=(4, 0)), np.empty(4)),
        (np.empty(shape=(5, 3)), np.empty(5)),
    ],
)
def test_init_state_hook_approx_threshold(
    x, y, chaos_fun, epsilon, state_fixture: ProcessingState
):
    x, x_counts = prepare_factorized_array(np.asarray(x))
    y, y_count = prepare_factorized_vector(np.asarray(y))
    state_fixture.values = {
        HOOKS_DATA_X: x,
        HOOKS_DATA_X_COUNTS: x_counts,
        HOOKS_DATA_Y: y,
        HOOKS_DATA_Y_COUNT: y_count,
    }
    state_fixture.config = {
        HOOKS_CHAOS_FUN: chaos_fun,
        HOOKS_EPSILON: epsilon,
    }
    init_state_hook_approx_threshold(state_fixture)
    base_chaos_score = get_chaos_score_for_data(
        x, x_counts, y, y_count, chaos_fun=chaos_fun, attrs=[]
    )
    total_chaos_score = get_chaos_score_for_data(
        x, x_counts, y, y_count, chaos_fun=chaos_fun, attrs=None
    )
    assert state_fixture.values[HOOKS_CHAOS_SCORE_BASE] == base_chaos_score
    assert state_fixture.values[HOOKS_CHAOS_SCORE_TOTAL] == total_chaos_score
    delta = (base_chaos_score - total_chaos_score) * epsilon
    approx_threshold = total_chaos_score + delta
    assert np.isclose(
        state_fixture.values[HOOKS_CHAOS_SCORE_APPROX_THRESHOLD], approx_threshold
    )
