from contextlib import nullcontext as does_not_raise
from typing import List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from skrough.algorithms.exceptions import LoopBreak
from skrough.algorithms.meta.aggregates import (
    InnerStopHooksAggregate,
    StopHooksAggregate,
    UpdateStateHooksAggregate,
)
from skrough.structs.state import ProcessingState

stop_hooks_parametrize: List[Tuple] = [
    (None, False, pytest.raises(ValueError, match="should not be empty")),
    (None, True, pytest.raises(ValueError, match="should not be empty")),
    ([], False, pytest.raises(ValueError, match="should not be empty")),
    ([], True, pytest.raises(ValueError, match="should not be empty")),
    (False, False, does_not_raise()),
    (False, True, does_not_raise()),
    (True, False, does_not_raise()),
    (True, True, pytest.raises(LoopBreak)),
    ([False], False, does_not_raise()),
    ([False], True, does_not_raise()),
    ([True], False, does_not_raise()),
    ([True], True, pytest.raises(LoopBreak)),
    ([False, False, True], False, does_not_raise()),
    ([False, False, True], True, pytest.raises(LoopBreak)),
    ([True, False, True], False, does_not_raise()),
    ([True, False, True], True, pytest.raises(LoopBreak)),
]


def prepare_stop_hook_mockup(mock, hook_values):
    if hook_values is None:
        hooks = None
        values = []
    elif isinstance(hook_values, bool):
        hooks = mock
        values = [hook_values]
    else:
        hooks = [mock for _ in range(len(hook_values))]
        values = hook_values

    # set side effects
    mock.side_effect = values

    return hooks, values


@pytest.mark.parametrize(
    "hook_values, raise_loop_break, exception_raise",
    stop_hooks_parametrize,
)
def test_stop_hooks_aggregate(
    hook_values,
    raise_loop_break,
    exception_raise,
    state_fixture: ProcessingState,
):
    mock = MagicMock()

    # let's handle None, One or a Sequence of hooks
    hooks: Optional[List[MagicMock]]
    values: List[bool]

    hooks, values = prepare_stop_hook_mockup(mock, hook_values)

    with exception_raise:
        agg_hooks = StopHooksAggregate.from_hooks(hooks)
        result = agg_hooks(state=state_fixture, raise_loop_break=raise_loop_break)
        assert result is any(values)
        # call count - it should be lazy and stop on first True
        expected_call_count = values.index(True) + 1 if True in values else len(values)
        assert mock.call_count == expected_call_count


@pytest.mark.parametrize(
    "hook_values, raise_loop_break, exception_raise",
    stop_hooks_parametrize,
)
def test_inner_stop_hooks_aggregate(
    hook_values,
    raise_loop_break,
    exception_raise,
    state_fixture: ProcessingState,
):
    mock = MagicMock()

    # let's handle None, One or a Sequence of hooks
    hooks: Optional[List[MagicMock]]
    values: List[bool]

    hooks, values = prepare_stop_hook_mockup(mock, hook_values)

    with exception_raise:
        agg_hooks = InnerStopHooksAggregate.from_hooks(hooks)
        result = agg_hooks(
            state=state_fixture,
            elements=[],
            raise_loop_break=raise_loop_break,
        )
        assert result is any(values)
        # call count - it should be lazy and stop on first True
        expected_call_count = values.index(True) + 1 if True in values else len(values)
        assert mock.call_count == expected_call_count


@pytest.mark.parametrize(
    "hook_values",
    [None, 0, [], [0], [0, 1, 2]],
)
def test_aggregate_update_state_hooks(
    hook_values,
    state_fixture: ProcessingState,
):
    mock = MagicMock()
    # let's handle None, One or a Sequence of hooks assuming that:
    # None ~ Optional (no hook)
    # a single int ~ One (a single hook)
    # a List ~ Sequence (multiple hooks)
    hooks: Optional[List[MagicMock]]
    values: List[int]

    if hook_values is None:
        hooks = None
        values = []
    elif isinstance(hook_values, int):
        hooks = mock
        values = [hook_values]
    else:
        hooks = [mock for _ in range(len(hook_values))]
        values = hook_values

    # set side effects
    mock.side_effect = values

    agg_hooks = UpdateStateHooksAggregate.from_hooks(hooks)
    agg_hooks(state=state_fixture)
    assert mock.call_count == len(values)
    for call in mock.call_args_list:
        assert call.args == (state_fixture,)
