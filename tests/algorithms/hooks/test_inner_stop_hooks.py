import numpy as np
import pytest

from skrough.algorithms.hooks.inner_stop_hooks import inner_stop_hook_empty
from skrough.structs.state import ProcessingState


@pytest.mark.parametrize(
    "elements, expected",
    [
        ([], True),
        (np.zeros(shape=0), True),
        ([0], False),
        ([1], False),
        ([1, 10, 0], False),
    ],
)
def test_inner_stop_hook_empty(
    elements,
    expected,
    state_fixture: ProcessingState,
):
    assert inner_stop_hook_empty(state_fixture, elements=elements) == expected
