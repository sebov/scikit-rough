from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from skrough.algorithms.hooks.finalize_hooks import finalize_hook_choose_objs_random
from skrough.algorithms.hooks.names import (
    VALUES_GROUP_INDEX,
    VALUES_RESULT_OBJS,
    VALUES_Y,
    VALUES_Y_COUNT,
)
from skrough.dataprep import prepare_factorized_vector
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import ProcessingState


@pytest.mark.parametrize(
    "group_index, y, permutation, expected_objs",
    [
        ([], [], [], []),
        ([0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]),
        ([0, 0, 0, 1], [0, 1, 0, 2], [0, 1, 2, 3], [0, 2, 3]),
        ([0, 0, 0, 1], [0, 1, 0, 2], [2, 1, 0, 3], [0, 2, 3]),
        ([0, 0, 0, 1], [0, 1, 0, 2], [1, 0, 2, 3], [1, 3]),
        ([0, 0, 1, 1], [0, 1, 0, 1], [0, 2, 1, 3], [0, 2]),
        ([0, 0, 1, 1], [0, 1, 0, 1], [0, 3, 1, 2], [0, 3]),
        ([0, 0, 1, 1], [0, 1, 0, 1], [1, 2, 0, 3], [1, 2]),
        ([0, 0, 1, 1], [0, 1, 0, 1], [1, 3, 0, 2], [1, 3]),
    ],
)
@patch("skrough.instances.get_permutation")
def test_finalize_state_hook_choose_objs_random(
    get_permutation_mock: MagicMock,
    group_index,
    y,
    permutation,
    expected_objs,
    state_fixture: ProcessingState,
):
    get_permutation_mock.return_value = np.asarray(permutation)

    group_index = GroupIndex.create_from_index(group_index)
    y, y_count = prepare_factorized_vector(y)
    state_fixture.values = {
        VALUES_GROUP_INDEX: group_index,
        VALUES_Y: y,
        VALUES_Y_COUNT: y_count,
    }
    finalize_hook_choose_objs_random(state_fixture)
    # is this a false positive unsubscriptable-object?
    # pylint: disable-next=unsubscriptable-object
    assert np.array_equal(state_fixture.values[VALUES_RESULT_OBJS], expected_objs)
