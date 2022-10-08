from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from skrough.algorithms.hooks.common.process_elements import (
    create_process_elements_hook_random_choice,
    process_elements_hook_pass_everything,
    process_elements_hook_reverse_elements,
)
from skrough.structs.state import ProcessingState
from tests.helpers import generate_data


def assert_rng_choice(rng_mock: MagicMock, elements, count):
    rng_mock.choice.assert_called_once()
    choice_call = rng_mock.choice.call_args
    choice_args = np.asarray(choice_call.args, dtype=object)
    expected_args = np.asarray((elements,), dtype=object)
    assert np.array_equal(choice_args, expected_args)
    assert choice_call.kwargs["size"] == count
    assert choice_call.kwargs["replace"] is False


def assert_draw_elements(rng_mock: MagicMock, elements, count, result):
    # the result should be unique as we assumed unique input elements
    assert len(np.unique(result)) == len(result)
    actual_count = min(len(elements), count)
    assert len(result) == actual_count
    assert np.isin(result, elements).all()
    assert_rng_choice(rng_mock, elements, actual_count)


@pytest.mark.parametrize(
    "config_key",
    ["foo", "bar"],
)
@pytest.mark.parametrize(
    "elements, max_count",
    [
        ([], None),
        ([], 0),
        ([], 1),
        ([], 10),
        ([0], None),
        ([0], 0),
        ([0], 1),
        ([0], 2),
        ([0], 10),
        ([0, 2], None),
        ([0, 2], 0),
        ([0, 2], 1),
        ([0, 2], 2),
        ([0, 2], 3),
        ([0, 2], 10),
        ([0, 1, 2, 3, 5], None),
        ([0, 1, 2, 3, 5], 0),
        ([0, 1, 2, 3, 5], 1),
        ([0, 1, 2, 3, 5], 5),
        ([0, 1, 2, 3, 5], 6),
        ([0, 1, 2, 3, 5], 10),
    ],
)
def test_(elements, max_count, config_key, state_fixture: ProcessingState):
    # for the test purpose let's assume unique input elements
    assert len(np.unique(elements)) == len(elements)

    hook_fun = create_process_elements_hook_random_choice(
        elements_count_config_key=config_key
    )
    rng_mock = cast(MagicMock, state_fixture.rng)
    # repeat several times as we test non deterministic function
    for _ in range(100):
        rng_mock.reset_mock()
        if max_count is None:
            # config:config_key not set -> random permutation of elements
            # elements
            result = hook_fun(state=state_fixture, elements=elements)
            assert_draw_elements(
                rng_mock=rng_mock, elements=elements, count=len(elements), result=result
            )
            # config:config_key set to None -> also random permutation of elements
            rng_mock.reset_mock()
            state_fixture.config = {config_key: None}
            result = hook_fun(state=state_fixture, elements=elements)
            assert_draw_elements(
                rng_mock=rng_mock, elements=elements, count=len(elements), result=result
            )
        else:
            state_fixture.config = {config_key: max_count}
            result = hook_fun(state=state_fixture, elements=elements)
            assert_draw_elements(
                rng_mock=rng_mock, elements=elements, count=max_count, result=result
            )


@pytest.mark.parametrize(
    "config_key",
    ["foo", "bar"],
)
@pytest.mark.parametrize(
    "elements, max_count",
    [
        ([0, 1, 1, 2, 3, 5], -1),
    ],
)
def test_process_elements_hook_random_choice_wrong_args(
    elements,
    max_count,
    config_key,
    state_fixture: ProcessingState,
):
    hook_fun = create_process_elements_hook_random_choice(
        elements_count_config_key=config_key
    )
    with pytest.raises(ValueError):
        state_fixture.config = {config_key: max_count}
        hook_fun(state_fixture, elements)


@pytest.mark.parametrize(
    "elements",
    [
        [],
        generate_data(size=0),
        [1],
        [-1],
        [0, 2],
        [1, 2, 3],
        np.arange(10),
    ],
)
def test_common_hook_pass_everything(
    elements,
    state_fixture: ProcessingState,
):
    result = process_elements_hook_pass_everything(state_fixture, elements)
    assert result is elements


@pytest.mark.parametrize(
    "elements",
    [
        [],
        generate_data(size=0),
        [1],
        [-1],
        [0, 2],
        [1, 2, 3],
        np.arange(10),
    ],
)
def test_common_hook_reverse_elements(
    elements,
    state_fixture: ProcessingState,
):
    result = process_elements_hook_reverse_elements(state_fixture, elements)
    assert np.array_equal(result, np.asarray(elements)[::-1])
