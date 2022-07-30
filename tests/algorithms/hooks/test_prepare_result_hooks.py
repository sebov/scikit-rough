import numpy as np
import pytest

from skrough.algorithms.hooks.names import HOOKS_RESULT_ATTRS, HOOKS_RESULT_OBJS
from skrough.algorithms.hooks.prepare_result_hooks import (
    prepare_result_hook_attrs_subset,
    prepare_result_hook_objs_attrs_subset,
)
from skrough.structs.attrs_subset import AttrsSubset
from skrough.structs.objs_attrs_subset import ObjsAttrsSubset
from skrough.structs.state import ProcessingState


@pytest.mark.parametrize(
    "result_attrs",
    [
        [],
        [0],
        [0, 1, 1, 2, 3, 5, 8],
        [0, 0, 1, 1, 2, 2, 9, 9],
    ],
)
def test_prepare_result_hook_attrs_subset(
    result_attrs,
    state_fixture: ProcessingState,
):
    for attrs in [result_attrs, np.asarray(result_attrs)]:
        state_fixture.values[HOOKS_RESULT_ATTRS] = attrs
        result = prepare_result_hook_attrs_subset(state_fixture)
        assert result == AttrsSubset.create_from(attrs_subset_like=result_attrs)


@pytest.mark.parametrize(
    "result_objs, result_attrs",
    [
        ([], []),
        ([], [0]),
        ([], [0, 1, 1, 2, 3, 5, 8]),
        ([], [0, 0, 1, 1, 2, 2, 9, 9]),
        ([0], []),
        ([0, 1, 1, 2, 3, 5, 8], []),
        ([0, 0, 1, 1, 2, 2, 9, 9], []),
        ([1], [4]),
        ([1, 2], [4, 1]),
        ([1, 1, 1, 2, 1], [5, 1, 1, 1, 4]),
    ],
)
def test_prepare_result_hook_objs_attrs_subset(
    result_objs,
    result_attrs,
    state_fixture: ProcessingState,
):
    for objs in [result_objs, np.asarray(result_objs)]:
        for attrs in [result_attrs, np.asarray(result_attrs)]:
            state_fixture.values[HOOKS_RESULT_OBJS] = objs
            state_fixture.values[HOOKS_RESULT_ATTRS] = attrs
            result = prepare_result_hook_objs_attrs_subset(state_fixture)
            assert result == ObjsAttrsSubset.create_from(
                objs_like=result_objs, attrs_like=result_attrs
            )
