from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from skrough.algorithms.exceptions import LoopBreak
from skrough.algorithms.hooks.pre_candidates_hooks import (
    pre_candidates_hook_remaining_attrs,
)
from skrough.structs.state import ProcessingState


@pytest.mark.parametrize(
    "x, result_attrs, exception_raise",
    [
        (np.eye(0), [], pytest.raises(LoopBreak, match="No remaining attrs")),
        (np.eye(5), [], does_not_raise()),
        (np.eye(5), [0, 1, 2], does_not_raise()),
        (
            np.eye(5),
            [0, 0, 0, 1, 2, 3, 4, 0],
            pytest.raises(LoopBreak, match="No remaining attrs"),
        ),
    ],
)
def test_pre_candidates_hook_remaining_attrs(
    x,
    result_attrs,
    exception_raise,
    state_fixture: ProcessingState,
):
    x = np.asarray(x)
    state_fixture.set_values_x(x)
    state_fixture.set_values_result_attrs(result_attrs)
    with exception_raise:
        result_pre_candidates = pre_candidates_hook_remaining_attrs(state_fixture)
        expected_pre_candidates = set(range(x.shape[1])) - set(result_attrs)
        assert set(result_pre_candidates) == expected_pre_candidates
