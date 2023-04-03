"""Feature importance functions."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union, cast

import joblib
import numpy as np
import pandas as pd

import skrough.typing as rght
from skrough.chaos_score import get_chaos_score_for_data
from skrough.structs.attrs_subset import AttrsSubset
from skrough.structs.objs_attrs_subset import ObjsAttrsSubset


@dataclass
class AttrsSubsetScoreGain:
    global_gain: rght.ChaosMeasureReturnType


AttrsSubsetScoreGainMapping = Dict[int, AttrsSubsetScoreGain]


@dataclass
class ObjsAttrsSubsetScoreGain:
    global_gain: rght.ChaosMeasureReturnType
    local_gain: rght.ChaosMeasureReturnType


ObjsAttrsSubsetScoreGainMapping = Dict[int, ObjsAttrsSubsetScoreGain]


FI_COLUMN_COL = "column"
FI_COUNT_COL = "count"

FI_GLOBAL_GAIN_COL = "global_gain"
FI_AVG_GLOBAL_GAIN_COL = "avg_global_gain"
FI_GLOBAL_GAIN_COVER_WEIGHTED_COL = "global_gain_cover_weighted"
FI_AVG_GLOBAL_GAIN_COVER_WEIGHTED_COL = "avg_global_gain_cover_weighted"

FI_LOCAL_GAIN_COL = "local_gain"
FI_AVG_LOCAL_GAIN_COL = "avg_local_gain"
FI_LOCAL_GAIN_COVER_WEIGHTED_COL = "local_gain_cover_weighted"
FI_AVG_LOCAL_GAIN_COVER_WEIGHTED_COL = "avg_local_gain_cover_weighted"


def _get_avg_over_counts(values, counts):
    result = np.true_divide(
        values,
        counts,
        out=np.zeros_like(values),
        where=counts > 0,
    )
    return result


# TODO: use the helper function also in compute_attrs_score_gains
def _get_chaos_score_for_data_multiple_input(
    xx_yy,
    x_counts,
    y_count,
    chaos_fun,
    attrs,
):
    return [
        get_chaos_score_for_data(
            x=xx,
            x_counts=x_counts,
            y=yy,
            y_count=y_count,
            chaos_fun=chaos_fun,
            attrs=attrs,
        )
        for (xx, yy) in xx_yy
    ]


def compute_attrs_score_gains(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    attrs_like: Union[AttrsSubset, rght.LocationsLike],
    chaos_fun: rght.ChaosMeasure,
) -> AttrsSubsetScoreGainMapping:
    """
    Compute feature importance for a single reduct
    """

    def _get_score(attrs_subset):
        (result,) = _get_chaos_score_for_data_multiple_input(
            xx_yy=[(x, y)],
            x_counts=x_counts,
            y_count=y_count,
            chaos_fun=chaos_fun,
            attrs=attrs_subset,
        )
        return result

    reduct = AttrsSubset.from_attrs_like(attrs_like)
    # let's prepare attrs concatenated with itself to apply sliding window approach
    # attrs_to_check = [a, b, c, d, a, b, c, d] ->
    #       get_chaos_score(..., attrs_to_check[1:4] <[b, c, d]>, ...)
    #       get_chaos_score(..., attrs_to_check[2:5] <[c, d, a]>, ...)
    #       get_chaos_score(..., attrs_to_check[3:6] <[d, a, b]>, ...)
    #       get_chaos_score(..., attrs_to_check[4:7] <[a, b, c]>, ...)
    attrs_to_check: Sequence[int] = reduct.attrs * 2
    attrs_len = len(reduct.attrs)
    result: AttrsSubsetScoreGainMapping = {}
    # unpack to 1-tuple just because reusing _get_chaos_score_for_data_multiple_input
    starting_chaos_score = _get_score(attrs_to_check[:attrs_len])
    for i in range(attrs_len):
        current_chaos_score = _get_score(
            attrs_to_check[(i + 1) : (i + attrs_len)],  # noqa: E203
        )
        result[attrs_to_check[i]] = AttrsSubsetScoreGain(
            global_gain=current_chaos_score - starting_chaos_score
        )
    return result


def get_feature_importance(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    column_names: Union[List[str], np.ndarray],
    attrs_subsets: Sequence[Union[AttrsSubset, rght.LocationsLike]],
    chaos_fun: rght.ChaosMeasure,
    n_jobs: Optional[int] = None,
):
    """
    Compute feature importance for a given collection of reducts
    """
    if x.shape[1] != len(column_names):
        raise ValueError("Data shape and column names mismatch")

    all_score_gains: Iterable[AttrsSubsetScoreGainMapping] = cast(
        Iterable[AttrsSubsetScoreGainMapping],
        joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(compute_attrs_score_gains)(
                x,
                x_counts,
                y,
                y_count,
                attrs_like,
                chaos_fun,
            )
            for attrs_like in attrs_subsets
        ),
    )

    counts = np.zeros(x.shape[1])
    global_gain = np.zeros(x.shape[1])
    for attrs_like, attr_score_gain_mapping in zip(attrs_subsets, all_score_gains):
        attrs_subset = AttrsSubset.from_attrs_like(attrs_like)
        counts[attrs_subset.attrs] += 1
        for attr in attrs_subset.attrs:
            global_gain[attr] += attr_score_gain_mapping[attr].global_gain
    result = pd.DataFrame(
        {
            FI_COLUMN_COL: column_names,
            FI_COUNT_COL: counts,
            FI_GLOBAL_GAIN_COL: global_gain,
            FI_AVG_GLOBAL_GAIN_COL: _get_avg_over_counts(global_gain, counts),
        }
    )
    return result


def compute_objs_attrs_score_gains(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    objs_attrs: ObjsAttrsSubset,
    chaos_fun: rght.ChaosMeasure,
) -> ObjsAttrsSubsetScoreGainMapping:
    """
    Compute feature importance for a single reduct
    """
    # let's prepare attrs concatenated with itself to apply sliding window approach
    # attrs_to_check = [a, b, c, d, a, b, c, d] ->
    #       get_chaos_score(..., attrs_to_check[1:4] <[b, c, d]>, ...)
    #       get_chaos_score(..., attrs_to_check[2:5] <[c, d, a]>, ...)
    #       get_chaos_score(..., attrs_to_check[3:6] <[d, a, b]>, ...)
    #       get_chaos_score(..., attrs_to_check[4:7] <[a, b, c]>, ...)
    attrs_to_check: Sequence[int] = objs_attrs.attrs * 2
    attrs_len = len(objs_attrs.attrs)
    result: ObjsAttrsSubsetScoreGainMapping = {}

    if len(objs_attrs.objs) == 0:
        result = {
            i: ObjsAttrsSubsetScoreGain(global_gain=0, local_gain=0)
            for i in range(attrs_len)
        }
        return result

    global_x = x
    global_y = y
    local_x = x[objs_attrs.objs]
    local_y = y[objs_attrs.objs]

    def _get_global_local_score(attrs_subset):
        result = _get_chaos_score_for_data_multiple_input(
            xx_yy=[(global_x, global_y), (local_x, local_y)],
            x_counts=x_counts,
            y_count=y_count,
            chaos_fun=chaos_fun,
            attrs=attrs_subset,
        )
        return result

    (
        global_starting_chaos_score,
        local_starting_chaos_score,
    ) = _get_global_local_score(attrs_to_check[:attrs_len])

    for i in range(attrs_len):
        (
            global_current_chaos_score,
            local_current_chaos_score,
        ) = _get_global_local_score(
            attrs_to_check[(i + 1) : (i + attrs_len)],  # noqa: E203
        )
        result[attrs_to_check[i]] = ObjsAttrsSubsetScoreGain(
            global_gain=global_current_chaos_score - global_starting_chaos_score,
            local_gain=local_current_chaos_score - local_starting_chaos_score,
        )
    return result


def get_feature_importance_for_objs_attrs(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    column_names: Union[List[str], np.ndarray],
    objs_attrs_collection: Sequence[ObjsAttrsSubset],
    chaos_fun: rght.ChaosMeasure,
    n_jobs: Optional[int] = None,
):
    """
    Compute feature importance for a given collection of bireducts
    """
    universe_len = x.shape[0]
    if universe_len == 0:
        raise ValueError("Data shape - no rows")
    if x.shape[1] != len(column_names):
        raise ValueError("Data shape and column names mismatch")

    score_gain_mappings_collection: Iterable[ObjsAttrsSubsetScoreGainMapping] = cast(
        Iterable[ObjsAttrsSubsetScoreGainMapping],
        joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(compute_objs_attrs_score_gains)(
                x,
                x_counts,
                y,
                y_count,
                objs_attrs,
                chaos_fun,
            )
            for objs_attrs in objs_attrs_collection
        ),
    )

    counts = np.zeros(x.shape[1])
    global_gain = np.zeros(x.shape[1])
    global_gain_cover_weighted = np.zeros(x.shape[1])
    local_gain = np.zeros(x.shape[1])
    local_gain_cover_weighted = np.zeros(x.shape[1])
    for objs_attrs, objs_attr_score_gain_mapping in zip(
        objs_attrs_collection, score_gain_mappings_collection
    ):
        counts[objs_attrs.attrs] += 1
        for attr in objs_attrs.attrs:
            global_gain_value = objs_attr_score_gain_mapping[attr].global_gain
            local_gain_value = objs_attr_score_gain_mapping[attr].local_gain
            cover_factor = len(objs_attrs.objs) / universe_len
            global_gain[attr] += global_gain_value
            global_gain_cover_weighted[attr] += global_gain_value * cover_factor
            local_gain[attr] += local_gain_value
            local_gain_cover_weighted[attr] += local_gain_value * cover_factor
    result = pd.DataFrame(
        {
            FI_COLUMN_COL: column_names,
            FI_COUNT_COL: counts,
            FI_GLOBAL_GAIN_COL: global_gain,
            FI_AVG_GLOBAL_GAIN_COL: _get_avg_over_counts(global_gain, counts),
            FI_GLOBAL_GAIN_COVER_WEIGHTED_COL: global_gain_cover_weighted,
            FI_AVG_GLOBAL_GAIN_COVER_WEIGHTED_COL: _get_avg_over_counts(
                global_gain_cover_weighted, counts
            ),
            FI_LOCAL_GAIN_COL: local_gain,
            FI_AVG_LOCAL_GAIN_COL: _get_avg_over_counts(local_gain, counts),
            FI_LOCAL_GAIN_COVER_WEIGHTED_COL: local_gain_cover_weighted,
            FI_AVG_LOCAL_GAIN_COVER_WEIGHTED_COL: _get_avg_over_counts(
                local_gain_cover_weighted, counts
            ),
        }
    )
    return result
