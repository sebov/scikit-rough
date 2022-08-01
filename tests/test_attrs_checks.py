import numpy as np
import pytest

from skrough.attrs_checks import check_if_attr_better_than_shuffled
from skrough.chaos_measures import conflicts_number, entropy, gini_impurity
from skrough.dataprep import prepare_factorized_vector
from skrough.structs.group_index import GroupIndex


@pytest.mark.flaky(max_runs=10)
@pytest.mark.parametrize(
    "chaos_fun",
    [
        conflicts_number,
        entropy,
        gini_impurity,
    ],
)
@pytest.mark.parametrize(
    "group_index, y, attr_values, n_of_probes, allowed_randomness, expected",
    [
        (
            [0, 0, 0, 1, 1, 1],
            [0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0],
            100,
            0.05,
            True,
        ),
        (
            [0, 0, 0, 1, 1, 1],
            [0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1],
            100,
            0.05,
            True,
        ),
        (
            [0, 0, 0, 1, 1, 1],
            [0, 1, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 1],
            100,
            0.25,
            False,
        ),
        (
            [0, 0, 0, 1, 1, 1],
            [0, 1, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 1],
            100,
            0.5,
            False,
        ),
    ],
)
def test_check_if_attr_better_than_shuffled(
    group_index,
    y,
    attr_values,
    n_of_probes,
    allowed_randomness,
    expected,
    chaos_fun,
):
    group_index = GroupIndex.create_from_index(group_index)
    attr_values, attr_values_count = prepare_factorized_vector(attr_values)
    y, y_count = prepare_factorized_vector(y)
    result = check_if_attr_better_than_shuffled(
        group_index=group_index,
        attr_values=attr_values,
        attr_values_count=attr_values_count,
        values=y,
        values_count=y_count,
        n_of_probes=n_of_probes,
        allowed_randomness=allowed_randomness,
        chaos_fun=chaos_fun,
        rng=np.random.default_rng(),
    )
    assert result is expected
