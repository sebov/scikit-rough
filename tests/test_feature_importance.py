import numpy as np
import pytest

import skrough as rgh


@pytest.mark.parametrize(
    "xx, xx_counts, yy, yy_count, column_names, reduct_list, expected",
    [
        (
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ],
            [2, 2],
            [0, 1, 0, 1],
            2,
            ["col1", "col2"],
            [[0], [0], [1]],
            [
                [2.0, 0.0],
                [1.0, 0.5],
            ],
        ),
    ],
)
def test_feature_importance(
    xx, xx_counts, yy, yy_count, column_names, reduct_list, expected
):
    xx = np.asarray(xx)
    xx_counts = np.asarray(xx_counts)
    yy = np.asarray(yy)
    result = rgh.feature_importance.get_feature_importance(
        xx,
        xx_counts,
        yy,
        yy_count,
        column_names,
        reduct_list,
        chaos_fun=rgh.measures.gini_impurity,
    )
    assert result["column"].to_list() == column_names
    assert np.array_equal(result[["count", "total_gain"]].values, np.asarray(expected))


def test_feature_importance_shape_mismatch():
    xx = np.asarray(
        [
            [0],
            [1],
        ]
    )
    with pytest.raises(ValueError):
        rgh.feature_importance.get_feature_importance(
            xx, None, None, None, ["col1", "col2"], [], None
        )
