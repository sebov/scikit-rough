import numpy as np
import pandas as pd
import pytest

import skrough as rgh
from skrough.dataprep import prepare_factorized_array, prepare_factorized_vector
from skrough.disorder_measures import gini_impurity
from skrough.structs.group_index import GroupIndex
from skrough.structs.objs_attrs_subset import ObjsAttrsSubset


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
        disorder_fun=rgh.disorder_measures.gini_impurity,
    )
    assert result["column"].to_list() == column_names
    assert np.array_equal(result[["count", "global_gain"]].values, np.asarray(expected))


def test_feature_importance_shape_mismatch():
    single_column_data = np.asarray(
        [
            [0],
            [1],
        ]
    )
    x, x_counts = prepare_factorized_array(single_column_data)
    y, y_count = prepare_factorized_vector(values=np.zeros(2))
    with pytest.raises(ValueError, match="Data shape and column names mismatch"):
        rgh.feature_importance.get_feature_importance(
            x=x,
            x_counts=x_counts,
            y=y,
            y_count=y_count,
            column_names=["col1", "col2"],
            attrs_subsets=[],
            disorder_fun=gini_impurity,
        )


# TODO: add tests
# --------------------------------------------------------------
# alternative implementation of feature importance for bireducts
# to be used in tests
# --------------------------------------------------------------


def get_disorder_score(x, x_count_distinct, y, y_count_distinct, attrs, disorder_fun):
    group_index = GroupIndex.from_data(x, x_count_distinct, list(attrs))
    return group_index.get_disorder_score(y, y_count_distinct, disorder_fun)


# pylint: disable-next=too-many-locals
def get_bireducts_scores(
    x, x_count, y, y_count, column_names, bireducts: list[ObjsAttrsSubset], disorder_fun
):
    counts = np.zeros(x.shape[1])
    global_gain = np.zeros(x.shape[1])
    global_gain_cover = np.zeros(x.shape[1])
    local_gain = np.zeros(x.shape[1])
    local_gain_cover = np.zeros(x.shape[1])
    for bireduct in bireducts:
        bireduct_objects = bireduct.objs
        bireduct_all_attrs = set(bireduct.attrs)
        global_x = x
        global_y = y
        local_x = x[bireduct_objects]
        local_y = y[bireduct_objects]
        global_starting_disorder_score = get_disorder_score(
            global_x,
            x_count,
            global_y,
            y_count,
            bireduct_all_attrs,
            disorder_fun,
        )
        local_starting_disorder_score = get_disorder_score(
            local_x,
            x_count,
            local_y,
            y_count,
            bireduct_all_attrs,
            disorder_fun,
        )
        counts[bireduct.attrs] += 1
        for attr in bireduct.attrs:
            attrs_to_check = bireduct_all_attrs.difference([attr])
            global_current_disorder_score = get_disorder_score(
                global_x,
                x_count,
                global_y,
                y_count,
                attrs_to_check,
                disorder_fun,
            )
            global_score_gain = (
                global_current_disorder_score - global_starting_disorder_score
            )
            global_gain[attr] += global_score_gain
            global_gain_cover[attr] += (
                global_score_gain * len(bireduct_objects) / x.shape[0]
            )
            local_current_disorder_score = get_disorder_score(
                local_x,
                x_count,
                local_y,
                y_count,
                attrs_to_check,
                disorder_fun,
            )
            local_score_gain = (
                local_current_disorder_score - local_starting_disorder_score
            )
            local_gain[attr] += local_score_gain
            local_gain_cover[attr] += (
                local_score_gain * len(bireduct_objects) / x.shape[0]
            )
    avg_global_gain = np.divide(
        global_gain, counts, out=np.zeros_like(global_gain), where=counts > 0
    )
    avg_global_gain_cover = np.divide(
        global_gain_cover,
        counts,
        out=np.zeros_like(global_gain_cover),
        where=counts > 0,
    )
    avg_local_gain = np.divide(
        local_gain, counts, out=np.zeros_like(local_gain), where=counts > 0
    )
    avg_local_gain_cover = np.divide(
        local_gain_cover, counts, out=np.zeros_like(local_gain_cover), where=counts > 0
    )
    result = pd.DataFrame(
        {
            "column": column_names,
            "count": counts,
            "global_gain": global_gain,
            "avg_global_gain": avg_global_gain,
            "global_gain_cover_weighted": global_gain_cover,
            "avg_global_gain_cover_weighted": avg_global_gain_cover,
            "local_gain": local_gain,
            "avg_local_gain": avg_local_gain,
            "local_gain_cover_weighted": local_gain_cover,
            "avg_local_gain_cover_weighted": avg_local_gain_cover,
        }
    )
    return result
