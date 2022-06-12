import numpy as np
import pytest

import skrough as rgh
from skrough.chaos_measures import gini_impurity
from skrough.dataprep import prepare_factorized_values, prepare_factorized_x


@pytest.mark.parametrize(
    "x, x_counts, y, y_count, column_names, reduct_list, expected",
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
    x, x_counts, y, y_count, column_names, reduct_list, expected
):
    x = np.asarray(x)
    x_counts = np.asarray(x_counts)
    y = np.asarray(y)
    result = rgh.feature_importance.get_feature_importance(
        x,
        x_counts,
        y,
        y_count,
        column_names,
        reduct_list,
        chaos_fun=rgh.chaos_measures.gini_impurity,
    )
    assert result["column"].to_list() == column_names
    assert np.array_equal(result[["count", "total_gain"]].values, np.asarray(expected))


def test_feature_importance_shape_mismatch():
    single_column_data = np.asarray(
        [
            [0],
            [1],
        ]
    )
    x, x_counts = prepare_factorized_x(single_column_data)
    y, y_count = prepare_factorized_values(values=np.zeros(2))
    with pytest.raises(ValueError):
        rgh.feature_importance.get_feature_importance(
            x=x,
            x_counts=x_counts,
            y=y,
            y_count=y_count,
            column_names=["col1", "col2"],
            attrs_subsets=[],
            chaos_fun=gini_impurity,
        )
