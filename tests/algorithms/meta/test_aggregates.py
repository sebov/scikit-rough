from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock

import numpy as np
import pytest

from skrough.algorithms.exceptions import LoopBreak
from skrough.algorithms.meta.aggregates import (
    ChainProcessElementsHooksAggregate,
    InnerStopHooksAggregate,
    ProcessElementsHooksAggregate,
    ProduceElementsHooksAggregate,
    StopHooksAggregate,
    UpdateStateHooksAggregate,
)
from skrough.structs.description_node import DescriptionNode
from skrough.structs.state import ProcessingState

stop_hooks_parametrize: list[tuple] = [
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
]


def prepare_hooks_mockup(mock, hook_values):
    """Prepare hooks mockup + values that will be returned by them."""
    if hook_values is None:
        hooks = None
        values = []
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

    hooks, values = prepare_hooks_mockup(mock, hook_values)

    # for stop-like aggregates we skip if ``hooks is None``
    if hooks is None:
        return

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

    hooks, values = prepare_hooks_mockup(mock, hook_values)

    # for stop-like aggregates we skip if ``hooks is None``
    if hooks is None:
        return

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
    [None, [], [0], [0, 1, 2]],
)
def test_update_state_hooks_aggregate(
    hook_values,
    state_fixture: ProcessingState,
):
    mock = MagicMock()

    hooks, values = prepare_hooks_mockup(mock, hook_values)

    agg_hooks = UpdateStateHooksAggregate.from_hooks(hooks)
    agg_hooks(state=state_fixture)
    assert mock.call_count == len(values)
    for call in mock.call_args_list:
        assert call.args == (state_fixture,)


produce_process_parametrize = [
    (None, []),
    ([(), ()], []),
    ([(0,), ()], [0]),
    ([(), (1,)], [1]),
    ([(0,), (1,)], [0, 1]),
    ([(0,), (), (), (), (0,)], [0]),
    ([(0, 1, 1), (0, 0, 1)], [0, 1]),
]


@pytest.mark.parametrize(
    "hook_values, expected",
    produce_process_parametrize,
)
def test_produce_elements_hooks_aggregate(
    hook_values,
    expected,
    state_fixture: ProcessingState,
):
    mock = MagicMock()

    hooks, values = prepare_hooks_mockup(mock, hook_values)

    agg_hooks = ProduceElementsHooksAggregate.from_hooks(hooks)
    result = agg_hooks(state=state_fixture)
    assert mock.call_count == len(values)
    for call in mock.call_args_list:
        assert call.args == (state_fixture,)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "hook_values, expected",
    produce_process_parametrize,
)
def test_process_elements_hooks_aggregate(
    hook_values,
    expected,
    state_fixture: ProcessingState,
):
    # let's prepare input elements argument that should be passed to each hook
    input_elements: tuple = (2, 7, 1, 8, 2, 8)

    mock = MagicMock()

    hooks, values = prepare_hooks_mockup(mock, hook_values)

    agg_hooks = ProcessElementsHooksAggregate.from_hooks(hooks)
    result = agg_hooks(state=state_fixture, elements=input_elements)
    assert mock.call_count == len(values)
    for call in mock.call_args_list:
        assert call.args == (state_fixture, input_elements)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "hook_values",
    [
        None,
        [(), ()],
        [(0,), ()],
        [(), (1,)],
        [(0,), (1,)],
        [(0,), (), (), (), (0,)],
        [(0, 1, 1), (0, 0, 1)],
        [(0, 0), (1, 1)],
    ],
)
def test_chain_process_elements_hooks_aggregate(
    hook_values,
    state_fixture: ProcessingState,
):
    # let's prepare input elements argument that should be passed to the first hook
    start_elements: tuple = (2, 7, 1, 8, 2, 8)

    mock = MagicMock()

    hooks, values = prepare_hooks_mockup(mock, hook_values)

    agg_hooks = ChainProcessElementsHooksAggregate.from_hooks(hooks)
    result = agg_hooks(state=state_fixture, elements=start_elements)
    assert mock.call_count == len(values)
    # everything starts with input_elements but later the return values from one hook is
    # passed as input to the next one
    # so, let's prepend values with start_elements and zip
    extended_values = [start_elements] + values
    for call, input_elements in zip(mock.call_args_list, extended_values[:-1]):
        assert call.args == (state_fixture, input_elements)
    assert np.array_equal(result, extended_values[-1])


@pytest.mark.parametrize(
    "agg_class, counts",
    [
        (StopHooksAggregate, [1, 2, 5]),
        (InnerStopHooksAggregate, [1, 2, 5]),
        (UpdateStateHooksAggregate, [None, 0, 1, 2, 5]),
        (ProduceElementsHooksAggregate, [None, 0, 1, 2, 5]),
        (ProcessElementsHooksAggregate, [None, 0, 1, 2, 5]),
        (ChainProcessElementsHooksAggregate, [None, 0, 1, 2, 5]),
    ],
)
def test_get_description_graph(agg_class, counts):
    mock = MagicMock()
    dummy_description_node = DescriptionNode(
        node_name="node_name",
        name="name",
        short_description="short_description",
        long_description="long_description",
    )
    mock.get_description_graph.return_value = dummy_description_node

    for count in counts:
        if count is None:
            hooks = None
            children_count = 0
        else:
            hooks = [mock for _ in range(count)]
            children_count = count

        aggregate = agg_class.from_hooks(hooks)
        result = aggregate.get_description_graph()

        # let us check the structure but ignore the description strings
        # therefore, we set them (short_description and long_description) for the
        # values that are actually in the result
        expected = DescriptionNode(
            name=agg_class.__name__,
            short_description=result.short_description,
            long_description=result.long_description,
            children=[dummy_description_node for _ in range(children_count)],
        )
        assert result == expected
