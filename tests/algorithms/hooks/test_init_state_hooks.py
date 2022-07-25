import numpy as np
import pytest

from skrough.algorithms.hooks.init_state_hooks import init_state_hook_single_group_index
from skrough.algorithms.hooks.names import HOOKS_DATA_X, HOOKS_GROUP_INDEX
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import ProcessingState


@pytest.mark.parametrize(
    "data",
    [
        np.empty(shape=(0, 0)),
        np.empty(shape=(0, 1)),
        np.empty(shape=(1, 0)),
        np.empty(shape=(2, 2)),
        np.empty(shape=(4, 1)),
        np.empty(shape=(4, 3)),
    ],
)
def test_init_state_hook_single_group_index(data, state_fixture: ProcessingState):
    state_fixture.values = {HOOKS_DATA_X: data}
    assert HOOKS_GROUP_INDEX not in state_fixture.values
    init_state_hook_single_group_index(state_fixture)
    assert HOOKS_GROUP_INDEX in state_fixture.values
    group_index: GroupIndex = state_fixture.values[HOOKS_GROUP_INDEX]
    assert group_index.n_objs == len(data)
    assert group_index.n_groups == (1 if len(data) > 0 else 0)
