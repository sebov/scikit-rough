from contextlib import nullcontext as does_not_raise
from typing import List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from skrough.algorithms.exceptions import LoopBreak
from skrough.algorithms.meta.helpers import (
    aggregate_any_inner_stop_hooks,
    aggregate_any_stop_hooks,
    aggregate_produce_elements_hooks,
    aggregate_update_state_hooks,
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
def test_aggregate_any_stop_hooks(
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
        agg_hooks = aggregate_any_stop_hooks(hooks)
        result = agg_hooks(state=state_fixture, raise_loop_break=raise_loop_break)
        assert result is any(values)
        # call count - it should be lazy and stop on first True
        expected_call_count = values.index(True) + 1 if True in values else len(values)
        assert mock.call_count == expected_call_count


@pytest.mark.parametrize(
    "hook_values, raise_loop_break, exception_raise",
    stop_hooks_parametrize,
)
def test_aggregate_any_inner_stop_hooks(
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
        agg_hooks = aggregate_any_inner_stop_hooks(hooks)
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

    agg_hooks = aggregate_update_state_hooks(hooks)
    agg_hooks(state_fixture)
    assert mock.call_count == len(values)
    for call in mock.call_args_list:
        assert call.args == (state_fixture,)


@pytest.mark.parametrize(
    "hook_values, expected",
    [
        (None, []),
        ((), []),
        ((1,), [1]),
        ((0, 1, 2, 5), [0, 1, 2, 5]),
        ((1, 1, 1, 1, 2, 1), [1, 2]),
        ((1, 1, 1, 1, 0, 1), [1, 0]),
        ([(), ()], []),
        ([(0,), ()], [0]),
        ([(), (1,)], [1]),
        ([(0,), (1,)], [0, 1]),
        ([(0,), (), (), (), (0,)], [0]),
        ([(0, 1, 1), (0, 0, 1)], [0, 1]),
    ],
)
def test_aggregate_produce_elements_hooks(
    hook_values,
    expected,
    state_fixture: ProcessingState,
):
    mock = MagicMock()
    # let's handle None, One or a Sequence of hooks assuming that:
    # None ~ Optional (no hook)
    # a List ~ One (a single hook)
    # a Tuple[List] ~ Sequence (multiple hooks)
    hooks: Optional[List[MagicMock]]
    values: List[Tuple]

    if hook_values is None:
        hooks = None
        values = []
    elif isinstance(hook_values, tuple):
        hooks = mock
        values = [hook_values]
    else:
        hooks = [mock for _ in range(len(hook_values))]
        values = hook_values

    # set side effects
    mock.side_effect = values

    agg_hooks = aggregate_produce_elements_hooks(hooks)
    result = agg_hooks(state_fixture)
    assert mock.call_count == len(values)
    for call in mock.call_args_list:
        assert call.args == (state_fixture,)
    assert np.array_equal(result, expected)
