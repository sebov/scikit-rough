from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def rng_mock():
    rng = MagicMock(wraps=np.random.default_rng())
    return rng
