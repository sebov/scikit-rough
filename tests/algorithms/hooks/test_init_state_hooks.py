import numpy as np
import pandas as pd
import pytest

from skrough.algorithms.hooks.init_hooks import (
    init_hook_approx_threshold,
    init_hook_factorize_data_x_y,
    init_hook_result_attrs_empty,
    init_hook_result_objs_empty,
    init_hook_single_group_index,
)
from skrough.algorithms.hooks.names import (
    CONFIG_CHAOS_FUN,
    CONFIG_EPSILON,
    INPUT_X,
    INPUT_Y,
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
from skrough.chaos_measures import conflicts_number, entropy, gini_impurity
from skrough.chaos_score import get_chaos_score_for_data
from skrough.dataprep import prepare_factorized_data
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import ProcessingState
from tests.algorithms.hooks.helpers import prepare_test_data_and_setup_state
from tests.helpers import generate_data


@pytest.mark.parametrize(
    "data",
    [
        [
            [0, 1],
            [1, 0],
        ],
        generate_data(size=(0, 1)),
        generate_data(size=(1, 1)),
        generate_data(size=(5, 1)),
        generate_data(size=(0, 2)),
        generate_data(size=(1, 2)),
        generate_data(size=(5, 2)),
        generate_data(size=(0, 5)),
        generate_data(size=(1, 5)),
        generate_data(size=(5, 5)),
    ],
)
def test_state_hook_factorize_data_x_y(data, state_fixture: ProcessingState):
    df = pd.DataFrame(data)
    x, x_counts, y, y_count = prepare_factorized_data(df, df.shape[1] - 1)

    state_fixture.input_data = {
        INPUT_X: df.iloc[:, :-1].to_numpy(),
        INPUT_Y: df.iloc[:, -1].to_numpy(),
    }
    init_hook_factorize_data_x_y(state_fixture)

    assert state_fixture.values.keys() == {
        VALUES_X,
        VALUES_X_COUNTS,
        VALUES_Y,
        VALUES_Y_COUNT,
    }
    assert np.array_equal(state_fixture.values[VALUES_X], x)
    assert np.array_equal(state_fixture.values[VALUES_X_COUNTS], x_counts)
    assert np.array_equal(state_fixture.values[VALUES_Y], y)
    assert np.array_equal(state_fixture.values[VALUES_Y_COUNT], y_count)


@pytest.mark.parametrize(
    "data",
    [
        generate_data(size=(0, 0)),
        generate_data(size=(0, 1)),
        generate_data(size=(1, 0)),
        generate_data(size=(2, 2)),
        generate_data(size=(4, 1)),
        generate_data(size=(4, 3)),
    ],
)
def test_init_hook_single_group_index(data, state_fixture: ProcessingState):
    state_fixture.values = {VALUES_X: data}
    assert VALUES_GROUP_INDEX not in state_fixture.values
    init_hook_single_group_index(state_fixture)
    assert VALUES_GROUP_INDEX in state_fixture.values
    group_index: GroupIndex = state_fixture.values[VALUES_GROUP_INDEX]
    assert group_index.n_objs == len(data)
    assert group_index.n_groups == (1 if len(data) > 0 else 0)


def test_init_hook_result_objs_empty(state_fixture: ProcessingState):
    state_fixture.values = {}
    init_hook_result_objs_empty(state_fixture)
    assert state_fixture.values == {VALUES_RESULT_OBJS: []}
    # let's invoke for the second time
    init_hook_result_objs_empty(state_fixture)
    assert state_fixture.values == {VALUES_RESULT_OBJS: []}


def test_init_hook_result_attrs_empty(state_fixture: ProcessingState):
    state_fixture.values = {}
    init_hook_result_attrs_empty(state_fixture)
    assert state_fixture.values == {VALUES_RESULT_ATTRS: []}
    # let's invoke for the second time
    init_hook_result_attrs_empty(state_fixture)
    assert state_fixture.values == {VALUES_RESULT_ATTRS: []}


@pytest.mark.parametrize("chaos_fun", [conflicts_number, entropy, gini_impurity])
@pytest.mark.parametrize("epsilon", [0.0, 0.1, 0.9, 1.0])
@pytest.mark.parametrize(
    "x, y",
    [
        ([[0], [1]], [0, 1]),
        (generate_data(size=(0, 0)), generate_data(size=0)),
        (generate_data(size=(0, 4)), generate_data(size=0)),
        (generate_data(size=(4, 0)), generate_data(size=4)),
        (generate_data(size=(5, 3)), generate_data(size=5)),
    ],
)
def test_init_hook_approx_threshold(
    x, y, chaos_fun, epsilon, state_fixture: ProcessingState
):
    state_fixture.config = {
        CONFIG_CHAOS_FUN: chaos_fun,
        CONFIG_EPSILON: epsilon,
    }
    x, x_counts, y, y_count, state_fixture = prepare_test_data_and_setup_state(
        x=x,
        y=y,
        state=state_fixture,
    )
    init_hook_approx_threshold(state_fixture)
    base_chaos_score = get_chaos_score_for_data(
        x, x_counts, y, y_count, chaos_fun=chaos_fun, attrs=[]
    )
    total_chaos_score = get_chaos_score_for_data(
        x, x_counts, y, y_count, chaos_fun=chaos_fun, attrs=None
    )
    assert state_fixture.values[VALUES_CHAOS_SCORE_BASE] == base_chaos_score
    assert state_fixture.values[VALUES_CHAOS_SCORE_TOTAL] == total_chaos_score
    delta = (base_chaos_score - total_chaos_score) * epsilon
    approx_threshold = total_chaos_score + delta
    assert np.isclose(
        state_fixture.values[VALUES_CHAOS_SCORE_APPROX_THRESHOLD], approx_threshold
    )
