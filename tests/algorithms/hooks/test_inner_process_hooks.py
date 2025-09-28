import numpy as np
import pytest

from skrough.algorithms.hooks.inner_process_hooks import (
    inner_process_hook_add_first_attr,
)
from skrough.dataprep import prepare_factorized_array
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import ProcessingState


@pytest.mark.parametrize(
    "data, attr_elements",
    [
        (np.eye(0), []),
        (np.eye(5), []),
        (np.eye(5), [0, 1, 2]),
        (np.eye(5), [0, 0, 0, 1, 2, 3, 4, 0]),
    ],
)
def test_inner_process_hook_add_first_attr(
    data,
    attr_elements,
    state_fixture: ProcessingState,
):
    x, x_counts = prepare_factorized_array(np.asarray(data))
    state_fixture.set_values_x(x)
    state_fixture.set_values_x_counts(x_counts)
    state_fixture.set_group_index(GroupIndex.create_uniform(len(x)))
    state_fixture.set_values_result_attrs([])

    expected_group_index = GroupIndex.create_uniform(len(x))
    expected_attrs = []

    # for tests let's loop |attrs_elements| times + (k=2) additional iterations
    # it is expected that attr_elements will shrink with each iteration until empty
    for _ in range(len(attr_elements) + 2):
        if len(attr_elements) > 0:
            attr = attr_elements[0]
            expected_group_index = expected_group_index.split(
                x[:, attr], x_counts[attr], compress=True
            )
            expected_attrs.append(attr)
        attr_elements = inner_process_hook_add_first_attr(
            state=state_fixture,
            elements=attr_elements,
        )
        assert state_fixture.get_values_result_attrs() == expected_attrs
        actual_group_index = state_fixture.get_group_index()
        assert actual_group_index.n_groups == expected_group_index.n_groups
        assert np.array_equal(actual_group_index.index, expected_group_index.index)
