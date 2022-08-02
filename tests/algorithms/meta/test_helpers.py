from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock

import pytest

from skrough.algorithms.exceptions import LoopBreak
from skrough.algorithms.meta.helpers import (
    aggregate_any_stop_hooks,
    normalize_hook_sequence,
)
from skrough.structs.state import ProcessingState

norm_mock = MagicMock()


@pytest.mark.parametrize(
    "hooks, optional, expected, exception_raise",
    [
        (None, False, None, pytest.raises(ValueError, match="should not be empty")),
        (None, True, [], does_not_raise()),
        ([], False, None, pytest.raises(ValueError, match="should not be empty")),
        ([], True, [], does_not_raise()),
        (norm_mock, False, [norm_mock], does_not_raise()),
        (norm_mock, True, [norm_mock], does_not_raise()),
        (norm_mock.another, False, [norm_mock.another], does_not_raise()),
        (norm_mock.another, True, [norm_mock.another], does_not_raise()),
        ([norm_mock.other], False, [norm_mock.other], does_not_raise()),
        ([norm_mock.other], True, [norm_mock.other], does_not_raise()),
        (
            [norm_mock.a1, norm_mock.a2, norm_mock.a3],
            False,
            [norm_mock.a1, norm_mock.a2, norm_mock.a3],
            does_not_raise(),
        ),
        (
            [norm_mock.a1, norm_mock.a2, norm_mock.a3],
            True,
            [norm_mock.a1, norm_mock.a2, norm_mock.a3],
            does_not_raise(),
        ),
        (
            [norm_mock.a1, norm_mock.a2, norm_mock.a3, norm_mock.a4, norm_mock.a5],
            False,
            [norm_mock.a1, norm_mock.a2, norm_mock.a3, norm_mock.a4, norm_mock.a5],
            does_not_raise(),
        ),
        (
            [norm_mock.a1, norm_mock.a2, norm_mock.a3, norm_mock.a4, norm_mock.a5],
            True,
            [norm_mock.a1, norm_mock.a2, norm_mock.a3, norm_mock.a4, norm_mock.a5],
            does_not_raise(),
        ),
    ],
)
def test_normalize_hook_sequence(hooks, optional, expected, exception_raise):
    with exception_raise:
        result = normalize_hook_sequence(hooks=hooks, optional=optional)
        assert result == expected


@pytest.mark.parametrize(
    "hook_values, raise_loop_break, exception_raise",
    [
        ([], False, pytest.raises(ValueError, match="should not be empty")),
        ([], True, pytest.raises(ValueError, match="should not be empty")),
        ([False], False, does_not_raise()),
        ([False], True, does_not_raise()),
        ([True], False, does_not_raise()),
        ([True], True, pytest.raises(LoopBreak)),
        ([False, False, True], False, does_not_raise()),
        ([False, False, True], True, pytest.raises(LoopBreak)),
        ([True, False, True], False, does_not_raise()),
        ([True, False, True], True, pytest.raises(LoopBreak)),
    ],
)
def test_aggregate_any_stop_hooks(
    hook_values,
    raise_loop_break,
    exception_raise,
    state_fixture: ProcessingState,
):
    mock = MagicMock()
    mock.side_effect = hook_values
    with exception_raise:
        agg_hook = aggregate_any_stop_hooks([mock for _ in range(len(hook_values))])
        result = agg_hook(state_fixture, raise_loop_break=raise_loop_break)
        assert result is any(hook_values)
        # call count - it should be lazy and stop on first True
        expected_call_count = (
            hook_values.index(True) + 1 if True in hook_values else len(hook_values)
        )
        assert mock.call_count == expected_call_count
