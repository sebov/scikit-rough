from typing import Dict, List, Optional, Sequence, cast

import joblib
import numpy as np
import pandas as pd

import skrough as rgh
import skrough.typing as rght
from skrough.containers import Reduct

ScoreGains = Dict[int, rght.ChaosMeasureReturnType]


def get_reduct_attrs(reduct: rght.ReductLike) -> List[int]:
    result = reduct.attrs if isinstance(reduct, Reduct) else list(reduct)
    return result


def compute_reduct_score_gains(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    reduct: rght.ReductLike,
    chaos_fun: rght.ChaosMeasure,
) -> ScoreGains:
    """
    Compute feature importance for a single reduct
    """
    reduct_attrs = get_reduct_attrs(reduct)
    attrs_to_check = reduct_attrs * 2
    attrs_len = len(reduct_attrs)
    score_gains: ScoreGains = {}
    starting_chaos_score = rgh.chaos_score.get_chaos_score(
        x, x_counts, y, y_count, attrs_to_check[:attrs_len], chaos_fun
    )
    for i in range(attrs_len):
        current_chaos_score = rgh.chaos_score.get_chaos_score(
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
    column_names: List[str],
    reducts: Sequence[rght.ReductLike],
    chaos_fun: rght.ChaosMeasure,
    n_jobs: Optional[int] = None,
):
    """
    Compute feature importance for a given collection of reducts
    """
    if x.shape[1] != len(column_names):
        raise ValueError("Data shape and column names mismatch.")

    score_gains_list: List[ScoreGains] = cast(
        List[ScoreGains],
        joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(compute_reduct_score_gains)(
                x,
                x_counts,
                y,
                y_count,
                reduct,
                chaos_fun,
            )
            for reduct in reducts
        ),
    )

    counts = np.zeros(x.shape[1])
    total_gain = np.zeros(x.shape[1])
    for reduct, score_gains in zip(reducts, score_gains_list):
        reduct_attrs = get_reduct_attrs(reduct)
        counts[reduct_attrs] += 1
        for attr in reduct_attrs:
            total_gain[attr] += score_gains[attr]
    avg_gain = np.true_divide(
        total_gain,
        counts,
        out=np.zeros_like(total_gain),
        where=counts > 0,
    )
    result = pd.DataFrame(
        {
            "column": column_names,
            "count": counts,
            "total_gain": total_gain,
            "avg_gain": avg_gain,
        }
    )
    return result
