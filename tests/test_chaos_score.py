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
    x, x_counts, y, y_count = rgh.dataprep.prepare_df(df, 3)
    return x, x_counts, y, y_count


@pytest.mark.parametrize(
    "chaos_fun",
    [
        rgh.measures.gini_impurity,
        rgh.measures.entropy,
    ],
)
@pytest.mark.parametrize(
    "attrs, expected_base",
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
def test_chaos_score(attrs, expected_base, chaos_fun, _test_data):
    result = rgh.chaos_score.compute_chaos_score(
        *_test_data, attrs=attrs, chaos_fun=chaos_fun
    )
    expected_base = np.asarray(expected_base)
    expected = chaos_fun(expected_base, expected_base.sum())
    assert result == expected
