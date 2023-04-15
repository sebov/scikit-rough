import json
import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import scipy
import xgboost as xgb

from skrough.algorithms.bireducts import get_bireduct_daab_heuristic
from skrough.chaos_measures import conflicts_count, entropy, gini_impurity
from skrough.feature_importance import get_feature_importance_for_objs_attrs

CHAOS_FUN_MAP = {
    "conflicts_count": conflicts_count,
    "entropy": entropy,
    "gini_impurity": gini_impurity,
}


@dataclass
class ParamsMixin:
    def asquery(self, key_prefix="run.params."):
        return " and ".join(
            f"{key_prefix}{k} == {json.dumps(v)}" for k, v in self.asdict().items()
        )

    def asdict(self):
        return asdict(self)


@dataclass
class BireductsParams(ParamsMixin):
    filename: str
    chaos_fun: str
    epsilon: float
    attrs_max_count: int
    candidates_count: int
    selected_count: int
    consecutive_daar_reps: int
    allowed_randomness: float
    probes_count: int
    n_bireducts: int


@dataclass
class XGBoostParams(ParamsMixin):
    filename: str
    num_boost_round: int
    learning_rate: float
    max_depth: int
    objective: str


def get_bireducts_scores(
    x,
    x_counts,
    y,
    y_count,
    column_names,
    chaos_fun,
    epsilon,
    attrs_max_count,
    candidates_count,
    selected_count,
    consecutive_daar_reps,
    allowed_randomness,
    probes_count,
    n_bireducts,
    seed,
    n_jobs,
):
    chaos_fun = CHAOS_FUN_MAP[chaos_fun]
    bireducts = get_bireduct_daab_heuristic(
        x,
        y,
        chaos_fun=chaos_fun,
        epsilon=epsilon,
        attrs_max_count=attrs_max_count,
        candidates_count=candidates_count,
        selected_count=selected_count,
        consecutive_daar_reps=consecutive_daar_reps,
        allowed_randomness=allowed_randomness,
        probes_count=probes_count,
        n_bireducts=n_bireducts,
        seed=seed,
        n_jobs=n_jobs,
    )
    bireducts_scores = get_feature_importance_for_objs_attrs(
        x=x,
        x_counts=x_counts,
        y=y,
        y_count=y_count,
        column_names=column_names,
        objs_attrs_collection=bireducts,
        chaos_fun=chaos_fun,
    )
    return bireducts_scores


def get_xgboost_scores(
    df,
    df_dec,
    num_boost_round,
    learning_rate,
    max_depth,
    objective,
):
    dec = df_dec.astype("category").cat.codes
    dtrain = xgb.DMatrix(df, label=dec)
    params = {
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "objective": objective,
        "num_class": dec.value_counts().size,
    }
    cl = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    result = pd.DataFrame({"column": df.columns})
    result.set_index("column", drop=False, inplace=True)
    for importance_type in ("weight", "gain", "cover", "total_gain", "total_cover"):
        result[importance_type] = 0
        score = cl.get_score(importance_type=importance_type)
        result.loc[score.keys(), importance_type] = list(score.values())
    result.reset_index(drop=True, inplace=True)
    return result


def get_correlation_scores(df, df_dec):
    result = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for column in df.columns:
            result.append([column] + list(scipy.stats.spearmanr(df[column], df_dec)))
    result = pd.DataFrame(result, columns=("column", "spearman_correlation", "pvalue"))
    idx = result["spearman_correlation"].isna()
    result.loc[idx, "spearman_correlation"] = 0
    result.loc[idx, "pvalue"] = 1
    result["spearman_correlation"] = result["spearman_correlation"].abs()
    result["pvalue_score"] = 1 / (result["pvalue"] + 10 ** (-20))
    return result


def latex_compare_result(compare_ranks_result):
    df = compare_ranks_result.pivot(
        index="top_k", columns="attr_type", values="avg_rank"
    )
    order = np.argsort([int(i) if i != "all" else 0 for i in df.index])
    df = df.iloc[order]
    print(df.style.to_latex())
