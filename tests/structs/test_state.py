import numpy as np
import pytest

from skrough.structs.state import (
    ProcessingState,
    StateConfig,
    StateValues,
)


def dummy_processing_fun(_: ProcessingState):
    """Just an empty processing function that does nothing."""


EMPTY_CONFIG: StateConfig = {}
EMPTY_VALUES: StateValues = {}


@pytest.mark.parametrize(
    "config, values, rng_seed",
    [
        (None, None, None),
        (None, None, 1),
        (None, {"a": 1}, None),
        (None, {"a": 1}, 1),
        ({"a": 1}, None, None),
        ({"a": 1}, None, 1),
        ({"a": 1}, {"a": 1}, None),
        ({"a": 1}, {"a": 1}, 1),
    ],
)
def test_state_from_optional(config, values, rng_seed):
    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed)
    else:
        rng = None
    state = ProcessingState.from_optional(
        rng=rng,
        processing_fun=dummy_processing_fun,
        config=config,
        values=values,
    )
    if config is None:
        assert state.config == EMPTY_CONFIG
    else:
        assert state.config is config

    if values is None:
        assert state.values == EMPTY_VALUES
    else:
        assert state.values is values

    if rng_seed is None:
        assert not state.is_set_rng()
    else:
        assert isinstance(state.get_rng(), np.random.Generator)
