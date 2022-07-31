from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from skrough.algorithms.hooks.candidates_hooks import candidates_hook_random_choice
from skrough.algorithms.hooks.names import HOOKS_CANDIDATES_MAX_COUNT
from skrough.structs.state import ProcessingState


def assert_rng_choice(rng_mock: MagicMock, elements, count):
    choice_call = rng_mock.choice.call_args
    choice_args = np.asarray(choice_call.args, dtype=object)
    expected_args = np.asarray((elements,), dtype=object)
    assert np.array_equal(choice_args, expected_args)
    assert choice_call.kwargs["size"] == count
    assert choice_call.kwargs["replace"] is False


def assert_all_elements(rng_mock: MagicMock, result, elements):
    assert np.array_equal(np.sort(result), np.sort(elements))
    assert_rng_choice(rng_mock, elements, len(elements))


def assert_draw_elements(rng_mock: MagicMock, count, result, elements):
    actual_count = min(len(elements), count)
    assert len(result) == actual_count
    assert np.isin(result, elements).all()
    assert_rng_choice(rng_mock, elements, actual_count)


@pytest.mark.parametrize(
    "elements, max_count",
    [
        ([], None),
        ([], 0),
        ([], 1),
        ([], 10),
        ([1], None),
        ([1], 0),
        ([1], 1),
        ([1], 10),
        ([0, 1, 1, 2, 3, 5], None),
        ([0, 1, 1, 2, 3, 5], 0),
        ([0, 1, 1, 2, 3, 5], 1),
        ([0, 1, 1, 2, 3, 5], 10),
    ],
)
def test_candidates_hook_random_choice(
    elements,
    max_count,
    state_fixture: ProcessingState,
):
    rng_mock = cast(MagicMock, state_fixture.rng)
    if max_count is None:
        # config:HOOKS_GROW_CANDIDATES_MAX_COUNT not set -> random choice from all
        # elements
        result = candidates_hook_random_choice(state_fixture, elements)
        assert_all_elements(rng_mock, result, elements)
        # config:HOOKS_GROW_CANDIDATES_MAX_COUNT set to None -> also random choice from
        # all elements
        rng_mock.reset_mock()
        state_fixture.config = {HOOKS_CANDIDATES_MAX_COUNT: None}
        result = candidates_hook_random_choice(state_fixture, elements)
        assert_all_elements(rng_mock, result, elements)
    else:
        state_fixture.config = {HOOKS_CANDIDATES_MAX_COUNT: max_count}
        result = candidates_hook_random_choice(state_fixture, elements)
        assert_draw_elements(rng_mock, max_count, result, elements)


@pytest.mark.parametrize(
    "elements, max_count",
    [
        ([0, 1, 1, 2, 3, 5], -1),
    ],
)
def test_candidates_hook_random_choice_wrong_args(
    elements,
    max_count,
    state_fixture: ProcessingState,
):
    with pytest.raises(ValueError):
        state_fixture.config = {HOOKS_CANDIDATES_MAX_COUNT: max_count}
        candidates_hook_random_choice(state_fixture, elements)
