from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from skrough.algorithms.reduct_ordering import get_reduct_ordering
from skrough.checks import check_if_reduct
from tests.helpers import generate_data


@pytest.mark.parametrize(
    "data, labels, permutation_size, exception_raise",
    [
        (
            generate_data(size=(0, 0)),
            generate_data(size=0),
            0,
            does_not_raise(),
        ),
        (
            generate_data(size=(0, 0)),
            generate_data(size=0),
            101,
            pytest.raises(
                ValueError,
                match="length of permutation should match the number of objects",
            ),
        ),
        (
            generate_data(size=(10, 0)),
            generate_data(size=10),
            0,
            does_not_raise(),
        ),
        (
            generate_data(size=(10, 0)),
            generate_data(size=10),
            101,
            pytest.raises(
                ValueError,
                match="length of permutation should match the number of objects",
            ),
        ),
        (
            generate_data(size=(0, 4)),
            generate_data(size=0),
            4,
            does_not_raise(),
        ),
        (
            generate_data(size=(0, 4)),
            generate_data(size=0),
            101,
            pytest.raises(
                ValueError,
                match="length of permutation should match the number of objects",
            ),
        ),
        (
            generate_data(size=(10, 4)),
            generate_data(size=10),
            4,
            does_not_raise(),
        ),
        (
            generate_data(size=(10, 4)),
            generate_data(size=10),
            101,
            pytest.raises(
                ValueError,
                match="length of permutation should match the number of objects",
            ),
        ),
        (
            generate_data(size=(10, 20), values_max=2),
            np.arange(10),
            20,
            does_not_raise(),
        ),
        (
            generate_data(size=(10, 1), values_max=2),
            generate_data(size=10, values_max=2),
            1,
            does_not_raise(),
        ),
    ],
)
def test_get_reduct_ordering(
    data,
    labels,
    permutation_size,
    exception_raise,
    rng_mock: np.random.Generator,
):
    with exception_raise:
        result = get_reduct_ordering(
            x=data,
            y=labels,
            permutation=rng_mock.permutation(permutation_size),
        )
        check = check_if_reduct(
            x=data,
            y=labels,
            attrs=result.attrs,
            consistent_table_check=False,
        )
        assert check is True
