import numpy as np
import pytest

from skrough.attrs_checks import check_if_attr_better_than_shuffled
from skrough.dataprep import prepare_factorized_vector
from skrough.disorder_measures import conflicts_count, entropy, gini_impurity
from skrough.structs.group_index import GroupIndex

TEST_SMOOTHING_PARAMETER = 1
TEST_FAST = False


@pytest.mark.flaky(max_runs=10)
@pytest.mark.parametrize(
    "disorder_fun",
    [
        conflicts_count,
        entropy,
        gini_impurity,
    ],
)
@pytest.mark.parametrize(
    "group_index, y, attr_values, probes_count, allowed_randomness, expected",
    [
        # attr_values not introducing any information - so, even the allowed_randomness
        # is extremely large, expected is False
        (
            [0, 0, 0, 1, 1, 1],
            [0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0],
            1000,
            0.99,
            False,
        ),
        (
            [0, 0, 0, 1, 1, 1],
            [0, 1, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 1],
            1000,
            0.99,
            False,
        ),
        # attr_values introducing some information but it is not "perfect" - so, the
        # following two tests differ with allowed_randomness - for very low value we
        # expect False, for a little bit higher we expect True
        (
            [0, 0, 0, 1, 1, 1],
            [0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1],
            1000,
            0.01,
            False,
        ),
        (
            [0, 0, 0, 1, 1, 1],
            [0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1],
            1000,
            0.15,
            True,
        ),
        # attr_values introducing only a little bit information and is quite "bad" - so,
        # the following two tests differ with allowed_randomness - for high value we
        # expect False, for extremely large value we expect True
        (
            [0, 0, 0, 1, 1, 1],
            [0, 1, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 1],
            1000,
            0.8,
            False,
        ),
        (
            [0, 0, 0, 1, 1, 1],
            [0, 1, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 1],
            1000,
            0.95,
            True,
        ),
    ],
)
def test_check_if_attr_better_than_shuffled(
    group_index,
    y,
    attr_values,
    probes_count,
    allowed_randomness,
    expected,
    disorder_fun,
):
    group_index = GroupIndex.from_index(group_index)
    attr_values, attr_values_count = prepare_factorized_vector(attr_values)
    y, y_count = prepare_factorized_vector(y)
    result = check_if_attr_better_than_shuffled(
        group_index=group_index,
        attr_values=attr_values,
        attr_values_count=attr_values_count,
        values=y,
        values_count=y_count,
        probes_count=probes_count,
        allowed_randomness=allowed_randomness,
        smoothing_parameter=TEST_SMOOTHING_PARAMETER,
        fast=TEST_FAST,
        disorder_fun=disorder_fun,
        rng=np.random.default_rng(),
    )
    assert result is expected
