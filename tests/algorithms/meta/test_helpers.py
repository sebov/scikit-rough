# pylint: disable=duplicate-code

from contextlib import nullcontext as does_not_raise
from typing import List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from skrough.algorithms.meta.helpers import (
    aggregate_chain_process_elements_hooks,
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
    "hook_values",
    [
        None,
        (),
        (1,),
        (0, 1, 2, 5),
        (1, 1, 1, 1, 2, 1),
        (1, 1, 1, 1, 0, 1),
        [(), ()],
        [(0,), ()],
        [(), (1,)],
        [(0,), (1,)],
        [(0,), (), (), (), (0,)],
        [(0, 1, 1), (0, 0, 1)],
        [(0, 0), (1, 1)],
    ],
)
def test_aggregate_chain_process_elements_hooks(
    hook_values,
    state_fixture: ProcessingState,
):
    # let's prepare input elements argument that should be passed to the first hook
    start_elements: Tuple = (2, 7, 1, 8, 2, 8)

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

    agg_hooks = aggregate_chain_process_elements_hooks(hooks)
    result = agg_hooks(state=state_fixture, elements=start_elements)
    assert mock.call_count == len(values)
    # everything starts with input_elements but later the return values from one hook is
    # passed as input to the next one
    # so, let's prepend values with start_elements and zip
    extended_values = [start_elements] + values
    for call, input_elements in zip(mock.call_args_list, extended_values[:-1]):
        assert call.args == (state_fixture, input_elements)
    assert np.array_equal(result, extended_values[-1])
