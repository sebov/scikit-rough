import numpy as np
import pytest

from skrough.algorithms.hooks.names import (
    CONFIG_CHAOS_FUN,
    CONFIG_CONSECUTIVE_EMPTY_ITERATIONS_MAX_COUNT,
    CONFIG_RESULT_ATTRS_MAX_COUNT,
    VALUES_CHAOS_SCORE_APPROX_THRESHOLD,
    VALUES_CONSECUTIVE_EMPTY_ITERATIONS_COUNT,
    VALUES_GROUP_INDEX,
    VALUES_RESULT_ATTRS,
    VALUES_Y,
    VALUES_Y_COUNT,
)
from skrough.algorithms.hooks.stop_hooks import (
    stop_hook_approx_threshold,
    stop_hook_attrs_count,
    stop_hook_empty_iterations,
)
from skrough.chaos_measures import conflicts_count, entropy, gini_impurity
from skrough.dataprep import prepare_factorized_array, prepare_factorized_vector
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import ProcessingState
from tests.helpers import generate_data


@pytest.mark.parametrize(
    "chaos_fun",
    [
        conflicts_count,
        gini_impurity,
        entropy,
    ],
)
@pytest.mark.parametrize(
    "x, y, start_attrs",
    [
        (generate_data(size=(0, 0)), [], []),
        (generate_data(size=(2, 2)), [0, 0], [0]),
        (generate_data(size=(2, 2)), [0, 0], [0, 1]),
        (generate_data(size=(2, 2)), [0, 1], [0]),
        (generate_data(size=(2, 2)), [0, 1], [0, 1]),
        (generate_data(size=(5, 10)), generate_data(size=5), []),
        (generate_data(size=(5, 10)), generate_data(size=5), [0, 1, 2]),
        (generate_data(size=(5, 10)), generate_data(size=5), [0, 1, 2, 3, 4]),
    ],
)
def test_stop_hook_approx_threshold(
    x,
    y,
    start_attrs,
    chaos_fun,
    state_fixture: ProcessingState,
):
    x, x_counts = prepare_factorized_array(np.asarray(x))
    y, y_count = prepare_factorized_vector(np.asarray(y))
    group_index = GroupIndex.from_data(x=x, x_counts=x_counts, attrs=start_attrs)
    state_fixture.config = {
        CONFIG_CHAOS_FUN: chaos_fun,
    }
    state_fixture.values = {
        VALUES_Y: y,
        VALUES_Y_COUNT: y_count,
        VALUES_GROUP_INDEX: group_index,
    }

    chaos_score = group_index.get_chaos_score(
        values=y,
        values_count=y_count,
        chaos_fun=chaos_fun,
    )

    state_fixture.values[VALUES_CHAOS_SCORE_APPROX_THRESHOLD] = chaos_score
    assert stop_hook_approx_threshold(state_fixture) is True

    approx_threshold_less = np.nextafter(chaos_score, -np.inf)
    state_fixture.values[VALUES_CHAOS_SCORE_APPROX_THRESHOLD] = approx_threshold_less
    assert stop_hook_approx_threshold(state_fixture) is False


@pytest.mark.parametrize(
    "attrs, attrs_max_count",
    [
        ([], 0),
        ([0], 3),
        ([0, 1], 3),
        ([0, 1, 2], 3),
        ([0, 1, 2, 3], 3),
    ],
)
def test_stop_hook_attrs_count(
    attrs,
    attrs_max_count,
    state_fixture: ProcessingState,
):
    state_fixture.config = {CONFIG_RESULT_ATTRS_MAX_COUNT: attrs_max_count}
    state_fixture.values = {VALUES_RESULT_ATTRS: attrs}
    result = stop_hook_attrs_count(state_fixture)
    expected = len(attrs) >= attrs_max_count
    assert result == expected


@pytest.mark.parametrize(
    "empty_iterations_count, config_max_count",
    [
        (0, 0),
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3),
        (4, 3),
    ],
)
def test_stop_hook_empty_iterations(
    empty_iterations_count,
    config_max_count,
    state_fixture: ProcessingState,
):
    state_fixture.config = {
        CONFIG_CONSECUTIVE_EMPTY_ITERATIONS_MAX_COUNT: config_max_count
    }
    state_fixture.values = {
        VALUES_CONSECUTIVE_EMPTY_ITERATIONS_COUNT: empty_iterations_count
    }
    result = stop_hook_empty_iterations(state_fixture)
    expected = empty_iterations_count >= config_max_count
    assert result == expected
