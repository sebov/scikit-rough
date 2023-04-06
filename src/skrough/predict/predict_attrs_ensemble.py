# pylint: disable=duplicate-code

from __future__ import annotations

from typing import Iterable

import numpy as np

import skrough.typing as rght
from skrough.predict.helpers import PredictStrategy, predict_ensemble
from skrough.predict.predict_attrs import predict_attrs
from skrough.structs.attrs_subset import AttrsSubset


def predict_attrs_ensemble(
    model_ensemble: Iterable[AttrsSubset],
    reference_data: np.ndarray,
    reference_data_y: np.ndarray,
    reference_data_y_count: int,
    predict_data: np.ndarray,
    return_proba: bool = False,
    strategy: PredictStrategy = "original_order",
    seed: rght.Seed = None,
    n_jobs: int | None = None,
):
    return predict_ensemble(
        model_predict_fun=predict_attrs,
        model_ensemble=model_ensemble,
        reference_data=reference_data,
        reference_data_y=reference_data_y,
        reference_data_y_count=reference_data_y_count,
        predict_data=predict_data,
        return_proba=return_proba,
        strategy=strategy,
        seed=seed,
        n_jobs=n_jobs,
    )

    # rng = np.random.default_rng(seed)
    # predictions_collection = joblib.Parallel(n_jobs=n_jobs)(
    #     joblib.delayed(predict_objs_attrs)(
    #         model=attrs,
    #         reference_data=reference_data,
    #         reference_data_y=reference_data_y,
    #         predict_data=predict_data,
    #         strategy=strategy,
    #         seed=rng.integers(RNG_INTEGERS_PARAM),
    #     )
    #     for attrs in model
    # )

    # result, counts = aggregate_predictions(
    #     n_objs=len(predict_data),
    #     n_classes=reference_data_y_count,
    #     predictions_collection=numba.typed.List(predictions_collection),
    # )

    # if not return_proba:
    #     result = get_predictions_from_proba(result, counts)

    # return result
