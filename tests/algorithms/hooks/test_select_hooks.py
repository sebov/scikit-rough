from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from skrough.algorithms.hooks.names import (
    CONFIG_CHAOS_FUN,
    CONFIG_SELECT_ATTRS_CHAOS_SCORE_BASED_MAX_COUNT,
    CONFIG_SELECT_RANDOM_MAX_COUNT,
    VALUES_GROUP_INDEX,
)
from skrough.algorithms.hooks.select_hooks import (
    select_hook_attrs_chaos_score_based,
    select_hook_random,
)
from skrough.chaos_measures import conflicts_number, entropy, gini_impurity
from skrough.chaos_score import get_chaos_score_for_data
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import ProcessingState
from tests.algorithms.hooks.helpers import prepare_test_data_and_setup_state


@pytest.mark.parametrize(
    "elements, count",
    [
        ([], 0),
        ([], 2),
        ([0], 0),
        ([0], 1),
        ([0], 2),
        ([0, 2], 0),
        ([0, 2], 1),
        ([0, 2], 2),
        ([0, 2], 3),
    ],
)
def test_select_hook_random(
    elements,
    count,
    state_fixture: ProcessingState,
):
    state_fixture.config = {CONFIG_SELECT_RANDOM_MAX_COUNT: count}
    # repeat several times as we test non deterministic function
    for _ in range(100):
        rng_mock = cast(MagicMock, state_fixture.rng)
        rng_mock.reset_mock()

        result = select_hook_random(state_fixture, elements=elements)
        assert len(result) == min(len(elements), count)
        assert np.isin(np.asarray(result), elements).all()
        # for the test purpose let's use unique input elements
        assert len(np.unique(elements)) == len(elements)
        # then the result should also be unique
        assert len(np.unique(result)) == len(result)

        # check choice call - should it be tested this way?
        rng_mock.choice.assert_called_once()
        choice_call = rng_mock.choice.call_args
        assert choice_call.kwargs["replace"] is False


@pytest.mark.parametrize(
    "chaos_fun",
    [
        conflicts_number,
        gini_impurity,
        entropy,
    ],
)
@pytest.mark.parametrize(
    "x, y, start_attrs, count",
    [
        (np.empty(shape=(0, 0)), [], [], 0),
        (np.empty(shape=(0, 0)), [], [], 10),
        (np.empty(shape=(2, 2)), [0, 1], [], 0),
        (np.empty(shape=(2, 2)), [0, 1], [], 1),
        (np.empty(shape=(2, 2)), [0, 1], [], 2),
        (np.empty(shape=(2, 2)), [0, 1], [], 10),
        (np.empty(shape=(5, 5)), [0, 1, 1, 1, 0], [], 0),
        (np.empty(shape=(5, 5)), [0, 1, 1, 1, 0], [], 1),
        (np.empty(shape=(5, 5)), [0, 1, 1, 1, 0], [], 2),
        (np.empty(shape=(5, 5)), [0, 1, 1, 1, 0], [], 3),
        (np.empty(shape=(5, 5)), [0, 1, 1, 1, 0], [], 10),
        (np.empty(shape=(5, 5)), [0, 1, 1, 1, 0], [0, 3], 0),
        (np.empty(shape=(5, 5)), [0, 1, 1, 1, 0], [0, 3], 1),
        (np.empty(shape=(5, 5)), [0, 1, 1, 1, 0], [0, 3], 2),
        (np.empty(shape=(5, 5)), [0, 1, 1, 1, 0], [0, 3], 3),
        (np.empty(shape=(5, 5)), [0, 1, 1, 1, 0], [0, 3], 10),
        (np.empty(shape=(5, 5)), [0, 1, 1, 1, 0], [0, 1, 2, 3, 4], 0),
        (np.empty(shape=(5, 5)), [0, 1, 1, 1, 0], [0, 1, 2, 3, 4], 1),
        (np.empty(shape=(5, 5)), [0, 1, 1, 1, 0], [0, 1, 2, 3, 4], 2),
        (np.empty(shape=(5, 5)), [0, 1, 1, 1, 0], [0, 1, 2, 3, 4], 3),
        (np.empty(shape=(5, 5)), [0, 1, 1, 1, 0], [0, 1, 2, 3, 4], 10),
    ],
)
def test_select_hook_chaos_score_based(
    x,
    y,
    start_attrs,
    count,
    chaos_fun,
    state_fixture: ProcessingState,
):
    state_fixture.config = {
        CONFIG_CHAOS_FUN: chaos_fun,
        CONFIG_SELECT_ATTRS_CHAOS_SCORE_BASED_MAX_COUNT: count,
    }
    x, x_counts, y, y_count, state_fixture = prepare_test_data_and_setup_state(
        x=x,
        y=y,
        state=state_fixture,
    )
    group_index = GroupIndex.create_from_data(x, x_counts, start_attrs)
    state_fixture.values[VALUES_GROUP_INDEX] = group_index
    n_attrs = x.shape[1]
    result = select_hook_attrs_chaos_score_based(state_fixture, range(n_attrs))
    expected_count = min(n_attrs, count)
    assert len(result) == expected_count
    scores = []
    for i in range(n_attrs):
        scores.append(
            get_chaos_score_for_data(
                x=x,
                x_counts=x_counts,
                y=y,
                y_count=y_count,
                chaos_fun=chaos_fun,
                attrs=start_attrs + [i],
            )
        )
    expected_idx = np.argsort(scores)[:expected_count]
    expected = np.arange(n_attrs)[expected_idx]
    assert np.array_equal(result, expected)
