from unittest.mock import MagicMock

import numpy as np
import pytest

from skrough.algorithms.hooks.candidates_hooks import candidates_hook_random_choice
from skrough.algorithms.hooks.names import HOOKS_CANDIDATES_MAX_COUNT
from skrough.structs.state import ProcessingState
from tests.algorithms.hooks.helpers import dummy_processing_fun


def assert_rng_choice(rng, elements, count):
    choice_call = rng.choice.call_args
    choice_args = np.asarray(choice_call.args, dtype=object)
    expected_args = np.asarray((elements,), dtype=object)
    assert np.array_equal(choice_args, expected_args)
    assert choice_call.kwargs == {"size": count, "replace": False}


def assert_all_elements(rng: MagicMock, result, elements):
    assert np.array_equal(np.sort(result), np.sort(elements))
    assert_rng_choice(rng, elements, len(elements))


def assert_draw_elements(rng: MagicMock, count, result, elements):
    actual_count = min(len(elements), count)
    assert len(result) == actual_count
    assert np.isin(result, elements).all()
    assert_rng_choice(rng, elements, actual_count)


@pytest.mark.parametrize(
    "elements, max_count",
    [
        ([], None),
        ([], 1),
        ([], 10),
        ([1], None),
        ([1], 1),
        ([1], 10),
        ([0, 1, 1, 2, 3, 5], None),
        ([0, 1, 1, 2, 3, 5], 1),
        ([0, 1, 1, 2, 3, 5], 10),
    ],
)
def test_candidates_hook_random_choice(elements, max_count):
    rng = MagicMock(wraps=np.random.default_rng())
    state = ProcessingState.create_from_optional(
        rng=rng,
        processing_fun=dummy_processing_fun,
    )
    if max_count is None:
        # config:HOOKS_GROW_CANDIDATES_MAX_COUNT not set -> random choice from all
        # elements
        result = candidates_hook_random_choice(state, elements)
        assert_all_elements(rng, result, elements)
        # config:HOOKS_GROW_CANDIDATES_MAX_COUNT set to None -> also random choice from
        # all elements
        rng.reset_mock()
        state.config = {HOOKS_CANDIDATES_MAX_COUNT: None}
        result = candidates_hook_random_choice(state, elements)
        assert_all_elements(rng, result, elements)
    else:
        state.config = {HOOKS_CANDIDATES_MAX_COUNT: max_count}
        result = candidates_hook_random_choice(state, elements)
        assert_draw_elements(rng, max_count, result, elements)


@pytest.mark.parametrize(
    "elements, max_count",
    [
        ([0, 1, 1, 2, 3, 5], -1),
    ],
)
def test_candidates_hook_random_choice_wrong_args(elements, max_count):
    with pytest.raises(ValueError):
        state = ProcessingState.create_from_optional(
            rng=np.random.default_rng(),
            processing_fun=dummy_processing_fun,
            config={HOOKS_CANDIDATES_MAX_COUNT: max_count},
        )
        candidates_hook_random_choice(state, elements)
