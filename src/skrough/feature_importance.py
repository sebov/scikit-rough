from typing import Dict, Iterable, List, Optional, Sequence, Union, cast

import joblib
import numpy as np
import pandas as pd

import skrough.typing as rght
from skrough.chaos_score import get_chaos_score
from skrough.structs.attrs_subset import AttrsSubset

ScoreGains = Dict[int, rght.ChaosMeasureReturnType]


FI_COLUMN_COL = "column"
FI_COUNT_COL = "count"
FI_TOTAL_GAIN_COL = "total_gain"
FI_AVG_GAIN_COL = "avg_gain"


def compute_attrs_score_gains(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    attrs_like: Union[AttrsSubset, rght.AttrsLike],
    chaos_fun: rght.ChaosMeasure,
) -> ScoreGains:
    """
    Compute feature importance for a single reduct
    """
    reduct = AttrsSubset.create_from(attrs_like)
    # let's prepare attrs concatenated with itself to apply sliding window approach
    # attrs_to_check = [a, b, c, d, a, b, c, d] ->
    #       get_chaos_score(..., attrs_to_check[1:4] <[b, c, d]>, ...)
    #       get_chaos_score(..., attrs_to_check[2:5] <[c, d, a]>, ...)
    #       get_chaos_score(..., attrs_to_check[3:6] <[d, a, b]>, ...)
    #       get_chaos_score(..., attrs_to_check[4:7] <[a, b, c]>, ...)
    attrs_to_check: Sequence[int] = reduct.attrs * 2
    attrs_len = len(reduct.attrs)
    score_gains: ScoreGains = {}
    starting_chaos_score = get_chaos_score(
        x, x_counts, y, y_count, attrs_to_check[:attrs_len], chaos_fun
    )
    for i in range(attrs_len):
        current_chaos_score = get_chaos_score(
            x,
            x_counts,
            y,
            y_count,
            attrs_to_check[(i + 1) : (i + attrs_len)],  # noqa: E203
            chaos_fun,
        )
        score_gains[attrs_to_check[i]] = current_chaos_score - starting_chaos_score
    return score_gains


def get_feature_importance(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    column_names: Union[List[str], np.ndarray],
    attrs_subsets: Sequence[Union[AttrsSubset, rght.AttrsLike]],
    chaos_fun: rght.ChaosMeasure,
    n_jobs: Optional[int] = None,
):
    """
    Compute feature importance for a given collection of reducts
    """
    if x.shape[1] != len(column_names):
        raise ValueError("Data shape and column names mismatch.")

    all_score_gains: Iterable[ScoreGains] = cast(
        Iterable[ScoreGains],
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
    total_gain = np.zeros(x.shape[1])
    for attrs_like, score_gains in zip(attrs_subsets, all_score_gains):
        attrs_subset = AttrsSubset.create_from(attrs_like)
        counts[attrs_subset.attrs] += 1
        for attr in attrs_subset.attrs:
            total_gain[attr] += score_gains[attr]
    avg_gain = np.true_divide(
        total_gain,
        counts,
        out=np.zeros_like(total_gain),
        where=counts > 0,
    )
    result = pd.DataFrame(
        {
            FI_COLUMN_COL: column_names,
            FI_COUNT_COL: counts,
            FI_TOTAL_GAIN_COL: total_gain,
            FI_AVG_GAIN_COL: avg_gain,
        }
    )
    return result
