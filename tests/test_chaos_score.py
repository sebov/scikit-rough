import numpy as np
import pandas as pd
import pytest

import skrough as rgh


@pytest.fixture(scope="session")
def _test_data():
    df = pd.DataFrame(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
        ]
    )
    x, x_counts, y, y_count = rgh.dataprep.prepare_factorized_data(df, target_attr=3)
    return x, x_counts, y, y_count


@pytest.mark.parametrize(
    "chaos_fun",
    [
        rgh.chaos_measures.gini_impurity,
        rgh.chaos_measures.entropy,
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
    ],
)
def test_get_chaos_score_for_data(attrs, expected_distribution, chaos_fun, _test_data):
    result = rgh.chaos_score.get_chaos_score_for_data(
        *_test_data, attrs=attrs, chaos_fun=chaos_fun
    )
    expected_distribution = np.asarray(expected_distribution)
    expected = chaos_fun(expected_distribution, expected_distribution.sum())
    assert result == expected
