import numpy as np
import pytest

from skrough.algorithms.hooks.filter_hooks import filter_hook_attrs_first_daar
from skrough.algorithms.key_names import (
    CONFIG_DAAR_ALLOWED_RANDOMNESS,
    CONFIG_DAAR_PROBES_COUNT,
    CONFIG_DISORDER_FUN,
    VALUES_GROUP_INDEX,
)
from skrough.disorder_measures import conflicts_count, entropy, gini_impurity
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import ProcessingState
from tests.algorithms.hooks.helpers import prepare_test_data_and_setup_state


# @pytest.mark.flaky(max_runs=10)
@pytest.mark.parametrize(
    "disorder_fun",
    [
        conflicts_count,
        entropy,
        gini_impurity,
    ],
)
@pytest.mark.parametrize(
    "x, y, start_attrs, daar_probes_count",
    [
        (
            [
                [0, 1],
                [1, 1],
                [0, 0],
            ],
            [1, 1, 0],
            [],
            1000,
        ),
    ],
)
@pytest.mark.parametrize(
    "daar_allowed_randomness, elements, expected",
    [
        (0.05, [], []),
        (1.0, [], []),
        # 0 attr is very bad attr
        (0.99, [0], []),
        (1.0, [0], [0]),
        # 1 attr is very good attr - but it is quite likely that random sample (as there
        # are 3 elements) will be exactly good as well
        (0.2, [1], []),
        (0.4, [1], [1]),
        # combination of the above - but we test the filter that leaves only the first
        # attr meeting the criteria
        (0.2, [0, 1], []),
        (0.4, [0, 1], [1]),
        (0.99, [0, 1], [1]),
        (1.0, [0, 1], [0]),
    ],
)
def test_filter_hook_attrs_first_daar(
    x,
    y,
    start_attrs,
    daar_probes_count,
    daar_allowed_randomness,
    disorder_fun,
    elements,
    expected,
    state_fixture: ProcessingState,
):
    state_fixture.config = {
        CONFIG_DAAR_ALLOWED_RANDOMNESS: daar_allowed_randomness,
        CONFIG_DAAR_PROBES_COUNT: daar_probes_count,
        CONFIG_DISORDER_FUN: disorder_fun,
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
