import numpy as np
import pytest

from skrough.algorithms.hooks.common_hooks import (
    common_hook_pass_everything,
    common_hook_reverse_elements,
)
from skrough.structs.state import ProcessingState


@pytest.mark.parametrize(
    "elements",
    [
        [],
        np.empty(shape=0),
        [1],
        [-1],
        [0, 2],
        [1, 2, 3],
        np.arange(10),
    ],
)
def test_common_hook_pass_everything(
    elements,
    state_fixture: ProcessingState,
):
    result = common_hook_pass_everything(state_fixture, elements)
    assert result is elements


@pytest.mark.parametrize(
    "elements",
    [
        [],
        np.empty(shape=0),
        [1],
        [-1],
        [0, 2],
        [1, 2, 3],
        np.arange(10),
    ],
)
def test_common_hook_reverse_elements(
    elements,
    state_fixture: ProcessingState,
):
    result = common_hook_reverse_elements(state_fixture, elements)
    assert np.array_equal(result, np.asarray(elements)[::-1])
