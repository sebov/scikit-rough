# pylint: disable=duplicate-code

import numpy as np
import pytest

from skrough.algorithms.hooks.select_hooks import select_hook_attrs_disorder_score_based
from skrough.disorder_measures import conflicts_count, entropy, gini_impurity
from skrough.disorder_score import get_disorder_score_for_data
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import ProcessingState
from tests.algorithms.hooks.helpers import prepare_test_data_and_setup_state
from tests.helpers import generate_data


@pytest.mark.parametrize(
    "disorder_fun",
    [
        conflicts_count,
        gini_impurity,
        entropy,
    ],
)
@pytest.mark.parametrize(
    "x, y, start_attrs, count",
    [
        (generate_data(size=(0, 0)), [], [], 0),
        (generate_data(size=(0, 0)), [], [], 10),
        (generate_data(size=(2, 2)), [0, 1], [], 0),
        (generate_data(size=(2, 2)), [0, 1], [], 1),
        (generate_data(size=(2, 2)), [0, 1], [], 2),
        (generate_data(size=(2, 2)), [0, 1], [], 10),
        (generate_data(size=(5, 5)), [0, 1, 1, 1, 0], [], 0),
        (generate_data(size=(5, 5)), [0, 1, 1, 1, 0], [], 1),
        (generate_data(size=(5, 5)), [0, 1, 1, 1, 0], [], 2),
        (generate_data(size=(5, 5)), [0, 1, 1, 1, 0], [], 3),
        (generate_data(size=(5, 5)), [0, 1, 1, 1, 0], [], 10),
        (generate_data(size=(5, 5)), [0, 1, 1, 1, 0], [0, 3], 0),
        (generate_data(size=(5, 5)), [0, 1, 1, 1, 0], [0, 3], 1),
        (generate_data(size=(5, 5)), [0, 1, 1, 1, 0], [0, 3], 2),
        (generate_data(size=(5, 5)), [0, 1, 1, 1, 0], [0, 3], 3),
        (generate_data(size=(5, 5)), [0, 1, 1, 1, 0], [0, 3], 10),
        (generate_data(size=(5, 5)), [0, 1, 1, 1, 0], [0, 1, 2, 3, 4], 0),
        (generate_data(size=(5, 5)), [0, 1, 1, 1, 0], [0, 1, 2, 3, 4], 1),
        (generate_data(size=(5, 5)), [0, 1, 1, 1, 0], [0, 1, 2, 3, 4], 2),
        (generate_data(size=(5, 5)), [0, 1, 1, 1, 0], [0, 1, 2, 3, 4], 3),
        (generate_data(size=(5, 5)), [0, 1, 1, 1, 0], [0, 1, 2, 3, 4], 10),
    ],
)
def test_select_hook_disorder_score_based(
    x,
    y,
    start_attrs,
    count,
    disorder_fun,
    state_fixture: ProcessingState,
):
    state_fixture.set_config_select_attrs_disorder_score_based_max_count(count)
    state_fixture.set_config_disorder_fun(disorder_fun)
    x, x_counts, y, y_count, state_fixture = prepare_test_data_and_setup_state(
        x=x,
        y=y,
        state=state_fixture,
    )
    group_index = GroupIndex.from_data(x, x_counts, start_attrs)
    state_fixture.set_values_group_index(group_index)
    n_attrs = x.shape[1]
    result = select_hook_attrs_disorder_score_based(state_fixture, range(n_attrs))
    expected_count = min(n_attrs, count)
    assert len(result) == expected_count
    scores = []
    for i in range(n_attrs):
        scores.append(
            get_disorder_score_for_data(
                x=x,
                x_counts=x_counts,
                y=y,
                y_count=y_count,
                disorder_fun=disorder_fun,
                attrs=start_attrs + [i],
            )
        )
    expected_idx = np.argsort(scores)[:expected_count]
    expected = np.arange(n_attrs)[expected_idx]
    assert np.array_equal(result, expected)
