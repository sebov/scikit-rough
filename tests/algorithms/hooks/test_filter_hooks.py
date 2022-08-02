import numpy as np
import pytest

from skrough.algorithms.hooks.filter_hooks import filter_hook_attrs_first_daar
from skrough.algorithms.hooks.names import (
    CONFIG_CHAOS_FUN,
    CONFIG_DAAR_ALLOWED_RANDOMNESS,
    CONFIG_DAAR_N_OF_PROBES,
    VALUES_GROUP_INDEX,
)
from skrough.chaos_measures import conflicts_number, entropy, gini_impurity
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import ProcessingState
from tests.algorithms.hooks.helpers import prepare_test_data_and_setup_state


@pytest.mark.flaky(max_runs=10)
@pytest.mark.parametrize(
    "chaos_fun",
    [
        conflicts_number,
        entropy,
        gini_impurity,
    ],
)
@pytest.mark.parametrize(
    "x, y, start_attrs, daar_n_of_probes",
    [
        (
            [
                [0, 1],
                [1, 1],
                [0, 0],
            ],
            [1, 1, 0],
            [],
            100,
        ),
    ],
)
@pytest.mark.parametrize(
    "daar_allowed_randomness, elements, expected",
    [
        (0.05, [], []),
        (1.0, [], []),
        (0.05, [0], []),
        (1.0, [0], [0]),
        (0.05, [1], [1]),
        (1.0, [1], [1]),
        (0.05, [0, 1], [1]),
        (1.0, [0, 1], [0]),
    ],
)
def test_filter_hook_attrs_first_daar(
    x,
    y,
    start_attrs,
    daar_n_of_probes,
    daar_allowed_randomness,
    chaos_fun,
    elements,
    expected,
    state_fixture: ProcessingState,
):
    state_fixture.config = {
        CONFIG_DAAR_ALLOWED_RANDOMNESS: daar_allowed_randomness,
        CONFIG_DAAR_N_OF_PROBES: daar_n_of_probes,
        CONFIG_CHAOS_FUN: chaos_fun,
    }
    x, x_counts, y, _, state_fixture = prepare_test_data_and_setup_state(
        x=x,
        y=y,
        state=state_fixture,
    )
    group_index = GroupIndex.from_data(x, x_counts, start_attrs)
    state_fixture.values[VALUES_GROUP_INDEX] = group_index
    result = filter_hook_attrs_first_daar(state_fixture, elements)
    assert np.array_equal(result, expected)
