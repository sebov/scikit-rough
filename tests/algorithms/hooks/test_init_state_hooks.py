import numpy as np
import pandas as pd
import pytest

from skrough.algorithms.hooks.init_hooks import (
    init_hook_epsilon_approx_threshold,
    init_hook_factorize_data_x_y,
    init_hook_result_attrs_empty,
    init_hook_result_objs_empty,
    init_hook_single_group_index,
)
from skrough.algorithms.key_names import (
    CONFIG_DISORDER_FUN,
    CONFIG_EPSILON,
    INPUT_DATA_X,
    INPUT_DATA_Y,
    VALUES_DISORDER_SCORE_APPROX_THRESHOLD,
    VALUES_DISORDER_SCORE_BASE,
    VALUES_DISORDER_SCORE_TOTAL,
    VALUES_RESULT_ATTRS,
    VALUES_X,
    VALUES_X_COUNTS,
)
from skrough.dataprep import prepare_factorized_data
from skrough.disorder_measures import conflicts_count, entropy, gini_impurity
from skrough.disorder_score import get_disorder_score_for_data
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
        INPUT_DATA_X: df.iloc[:, :-1].to_numpy(),
        INPUT_DATA_Y: df.iloc[:, -1].to_numpy(),
    }
    init_hook_factorize_data_x_y(state_fixture)

    assert state_fixture.values.keys() == {
        VALUES_X,
        VALUES_X_COUNTS,
    }
    assert np.array_equal(state_fixture.values[VALUES_X], x)
    assert np.array_equal(state_fixture.values[VALUES_X_COUNTS], x_counts)
    assert np.array_equal(state_fixture.get_values_y(), y)
    assert np.array_equal(state_fixture.get_values_y_count(), y_count)


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
    assert not state_fixture.is_set_group_index()
    init_hook_single_group_index(state_fixture)
    assert state_fixture.is_set_group_index()
    group_index = state_fixture.get_group_index()
    assert group_index.n_objs == len(data)
    assert group_index.n_groups == (1 if len(data) > 0 else 0)


def test_init_hook_result_objs_empty(state_fixture: ProcessingState):
    state_fixture.values = {}
    init_hook_result_objs_empty(state_fixture)
    assert state_fixture.get_values_result_objs() == []
    # let's invoke for the second time
    init_hook_result_objs_empty(state_fixture)
    assert state_fixture.get_values_result_objs() == []


def test_init_hook_result_attrs_empty(state_fixture: ProcessingState):
    state_fixture.values = {}
    init_hook_result_attrs_empty(state_fixture)
    assert state_fixture.values == {VALUES_RESULT_ATTRS: []}
    # let's invoke for the second time
    init_hook_result_attrs_empty(state_fixture)
    assert state_fixture.values == {VALUES_RESULT_ATTRS: []}


@pytest.mark.parametrize("disorder_fun", [conflicts_count, entropy, gini_impurity])
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
    x, y, disorder_fun, epsilon, state_fixture: ProcessingState
):
    state_fixture.config = {
        CONFIG_DISORDER_FUN: disorder_fun,
        CONFIG_EPSILON: epsilon,
    }
    x, x_counts, y, y_count, state_fixture = prepare_test_data_and_setup_state(
        x=x,
        y=y,
        state=state_fixture,
    )
    init_hook_epsilon_approx_threshold(state_fixture)
    base_disorder_score = get_disorder_score_for_data(
        x, x_counts, y, y_count, disorder_fun=disorder_fun, attrs=[]
    )
    total_disorder_score = get_disorder_score_for_data(
        x, x_counts, y, y_count, disorder_fun=disorder_fun, attrs=None
    )
    assert state_fixture.values[VALUES_DISORDER_SCORE_BASE] == base_disorder_score
    assert state_fixture.values[VALUES_DISORDER_SCORE_TOTAL] == total_disorder_score
    delta = (base_disorder_score - total_disorder_score) * epsilon
    approx_threshold = total_disorder_score + delta
    assert np.isclose(
        state_fixture.values[VALUES_DISORDER_SCORE_APPROX_THRESHOLD], approx_threshold
    )
