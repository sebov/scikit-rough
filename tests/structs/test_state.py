import numpy as np
import pytest

from skrough.structs.state import (
    ProcessingState,
    StateConfig,
)


def dummy_processing_fun(_: ProcessingState):
    """Just an empty processing function that does nothing."""


EMPTY_CONFIG: StateConfig = {}


@pytest.mark.parametrize(
    "config, rng_seed",
    [
        (None, None),
        (None, 1),
        ({"a": 1}, None),
        ({"a": 1}, 1),
    ],
)
def test_state_from_optional(config, rng_seed):
    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed)
    else:
        rng = None
    state = ProcessingState.from_optional(
        rng=rng,
        processing_fun=dummy_processing_fun,
        config=config,
    )
    if config is None:
        assert state.config == EMPTY_CONFIG
    else:
        assert state.config is config
    if rng_seed is None:
        assert not state.is_set_rng()
    else:
        assert isinstance(state.get_rng(), np.random.Generator)
