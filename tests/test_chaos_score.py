import numpy as np
import pandas as pd
import pytest

from skrough.chaos_measures import conflicts_number, entropy, gini_impurity
from skrough.chaos_score import get_chaos_score_for_data
from skrough.dataprep import prepare_factorized_data


@pytest.fixture(name="test_data", scope="session")
def fixture_test_data():
    df = pd.DataFrame(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
        ]
    )
    x, x_counts, y, y_count = prepare_factorized_data(df, target_attr=3)
    return x, x_counts, y, y_count


@pytest.mark.parametrize(
    "chaos_fun",
    [
        conflicts_number,
        entropy,
        gini_impurity,
    ],
)
@pytest.mark.parametrize(
    "attrs, expected_distribution",
    [
        (
            [0],
            [[1, 3]],
        ),
        (
            [1],
            [
                [1, 1],
                [0, 2],
            ],
        ),
        (
            [2],
            [
                [1, 0],
                [0, 3],
            ],
        ),
        (
            [0, 1],
            [
                [1, 1],
                [0, 2],
            ],
        ),
        (
            [0, 1, 2],
            [
                [1, 0],
                [0, 1],
                [0, 2],
            ],
        ),
        (
            None,
            [
                [1, 0],
                [0, 1],
                [0, 2],
            ],
        ),
    ],
)
def test_get_chaos_score_for_data(attrs, expected_distribution, chaos_fun, test_data):
    result = get_chaos_score_for_data(*test_data, chaos_fun=chaos_fun, attrs=attrs)
    expected_distribution = np.asarray(expected_distribution)
    expected = chaos_fun(expected_distribution, expected_distribution.sum())
    assert result == expected
