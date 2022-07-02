import numpy as np
import pytest

from skrough.structs.state import (
    ProcessingState,
    StateConfig,
    StateInputData,
    StateValues,
)


def dummy_processing_fun(_: ProcessingState):
    pass


EMPTY_CONFIG: StateConfig = {}
EMPTY_INPUT_DATA: StateInputData = {}
EMPTY_VALUES: StateValues = {}


@pytest.mark.parametrize(
    "config, input_data, values",
    [
        (None, None, None),
        ({"a": 1}, None, None),
        (None, {"a": 1}, None),
        (None, None, {"a": 1}),
        ({"a": 1}, {"a": 1}, None),
        ({"a": 1}, None, {"a": 1}),
        (None, {"a": 1}, {"a": 1}),
        ({"a": 1}, {"a": 1}, {"a": 1}),
    ],
)
def test_state_create_from_optional(config, input_data, values):
    rng = np.random.default_rng()
    state = ProcessingState.create_from_optional(
        rng=rng,
        processing_fun=dummy_processing_fun,
        config=config,
        input_data=input_data,
        values=values,
    )
    if config is None:
        assert state.config == EMPTY_CONFIG
    else:
        assert state.config is config

    if input_data is None:
        assert state.input_data == EMPTY_INPUT_DATA
    else:
        assert state.input_data is input_data

    if values is None:
        assert state.values == EMPTY_VALUES
    else:
        assert state.values is values
