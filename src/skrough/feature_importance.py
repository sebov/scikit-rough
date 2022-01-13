import joblib
import numpy as np
import pandas as pd

import skrough.chaos_score

# def get_feature_importance_slower(
#     xx,
#     xx_count_distinct,
#     yy,
#     yy_count_distinct,
#     column_names,
#     reduct_list,
#     chaos_fun,
#     _get_chaos_score_fun=skrough.chaos_score.get_chaos_score,
# ):
#     assert xx.shape[1] == len(column_names)

#     counts = np.zeros(xx.shape[1])
#     total_gain = np.zeros(xx.shape[1])
#     for reduct in reduct_list:
#         reduct = list(reduct)
#         reduct_all_attrs = set(reduct)
#         starting_chaos_score = _get_chaos_score_fun(
#             xx, xx_count_distinct, yy, yy_count_distinct, reduct_all_attrs, chaos_fun
#         )
#         counts[reduct] += 1
#         for attr in reduct:
#             attrs_to_check = reduct_all_attrs.difference([attr])
#             current_chaos_score = _get_chaos_score_fun(
#                 xx,
#                 xx_count_distinct,
#                 yy,
#                 yy_count_distinct,
#                 attrs_to_check,
#                 chaos_fun
#             )
#             score_gain = current_chaos_score - starting_chaos_score
#             total_gain[attr] += score_gain
#     avg_gain = np.divide(
#         total_gain, counts, out=np.zeros_like(total_gain), where=counts > 0
#     )
#     result = pd.DataFrame(
#         {
#             "column": column_names,
#             "count": counts,
#             "total_gain": total_gain,
#             "avg_gain": avg_gain,
#         }
#     )
#     return result


def compute_reduct_score_gains(
    xx,
    xx_count_distinct,
    yy,
    yy_count_distinct,
    reduct,
    chaos_fun,
    _get_chaos_score_fun,
):
    """
    Compute feature importance for a single reduct
    """
    score_gains = {}
    reduct = list(reduct)
    reduct_all_attrs = set(reduct)
    starting_chaos_score = _get_chaos_score_fun(
        xx, xx_count_distinct, yy, yy_count_distinct, reduct_all_attrs, chaos_fun
    )
    for attr in reduct:
        attrs_to_check = reduct_all_attrs.difference([attr])
        current_chaos_score = _get_chaos_score_fun(
            xx, xx_count_distinct, yy, yy_count_distinct, attrs_to_check, chaos_fun
        )
        score_gains[attr] = current_chaos_score - starting_chaos_score
    return score_gains


def get_feature_importance(
    xx,
    xx_count_distinct,
    yy,
    yy_count_distinct,
    column_names,
    reduct_list,
    chaos_fun,
    n_jobs=None,
):
    """
    Compute feature importance for a given collection of reducts
    """
    if xx.shape[1] != len(column_names):
        raise ValueError("Data shape and column names mismatch.")

    score_gains_list = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(compute_reduct_score_gains)(
            xx,
            xx_count_distinct,
            yy,
            yy_count_distinct,
            reduct,
            chaos_fun,
            _get_chaos_score_fun=skrough.chaos_score.compute_chaos_score,
        )
        for reduct in reduct_list
    )

    counts = np.zeros(xx.shape[1])
    total_gain = np.zeros(xx.shape[1])
    for reduct, score_gains in zip(reduct_list, score_gains_list):  # type: ignore
        reduct = list(reduct)
        counts[reduct] += 1
        for attr in reduct:
            total_gain[attr] += score_gains[attr]
    avg_gain = np.true_divide(
        total_gain,
        counts,
        out=np.zeros_like(total_gain),  # type: ignore
        where=counts > 0,
    )  # type: ignore
    result = pd.DataFrame(
        {
            "column": column_names,
            "count": counts,
            "total_gain": total_gain,
            "avg_gain": avg_gain,
        }
    )
    return result
