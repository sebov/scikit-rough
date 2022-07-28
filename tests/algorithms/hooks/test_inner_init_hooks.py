import numpy as np
import pytest

from skrough.algorithms.hooks.inner_init_hooks import (
    inner_init_hook_consecutive_empty_iterations_count,
)
from skrough.algorithms.hooks.names import HOOKS_CONSECUTIVE_EMPTY_ITERATIONS_COUNT
from skrough.structs.state import ProcessingState


@pytest.mark.parametrize(
    "elements_lengths, empty_iterations_counts",
    [
        ([0, 0], [1, 2]),
        ([0, 0, 0], [1, 2, 3]),
        ([3, 4], [0, 0]),
        ([1, 0, 0], [0, 1, 2]),
        ([1, 0, 1], [0, 1, 0]),
        ([1, 0, 0, 1, 10, 2, 0], [0, 1, 2, 0, 0, 0, 1]),
        ([0, 0, 5, 3, 2, 0, 0, 5, 3, 2], [1, 2, 0, 0, 0, 1, 2, 0, 0, 0]),
    ],
)
def test_inner_init_hook_consecutive_empty_iterations_count(
    elements_lengths,
    empty_iterations_counts,
    state_fixture: ProcessingState,
):
    assert HOOKS_CONSECUTIVE_EMPTY_ITERATIONS_COUNT not in state_fixture.values

    for elements_len, empty_count in zip(elements_lengths, empty_iterations_counts):
        inner_init_hook_consecutive_empty_iterations_count(
            state=state_fixture,
            elements=np.empty(shape=elements_len),
        )
        actual_empty_count = state_fixture.values.get(
            HOOKS_CONSECUTIVE_EMPTY_ITERATIONS_COUNT, 0
        )
        assert actual_empty_count == empty_count
