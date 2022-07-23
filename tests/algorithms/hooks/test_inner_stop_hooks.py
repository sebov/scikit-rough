import numpy as np
import pytest

from skrough.algorithms.hooks.inner_stop_hooks import inner_stop_hook_empty
from skrough.structs.state import ProcessingState


def dummy_processing_fun(_: ProcessingState):
    """Do nothing function of ProcessingFunction type."""


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
def test_inner_stop_hook_empty(elements, expected):
    state = ProcessingState.create_from_optional(
        rng=np.random.default_rng(),
        processing_fun=dummy_processing_fun,
    )
    assert inner_stop_hook_empty(state, elements=elements) == expected
