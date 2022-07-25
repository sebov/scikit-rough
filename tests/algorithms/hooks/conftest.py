from unittest.mock import MagicMock

import numpy as np
import pytest

from skrough.structs.state import ProcessingState


def dummy_processing_fun(_: ProcessingState):
    """Do nothing function of ProcessingFunction type."""


@pytest.fixture(name="rng_mock")
def fixture_rng_mock():
    rng = MagicMock(wraps=np.random.default_rng())
    return rng


@pytest.fixture
def state_fixture(rng_mock):
    state = ProcessingState.create_from_optional(
        rng=rng_mock,
        processing_fun=dummy_processing_fun,
    )
    return state
