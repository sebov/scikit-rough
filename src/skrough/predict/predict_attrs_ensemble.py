# pylint: disable=duplicate-code

from __future__ import annotations

from typing import Iterable

import numpy as np

import skrough.typing as rght
from skrough.predict.helpers import (
    NoAnswerStrategyKey,
    PredictStrategyKey,
    predict_ensemble,
)
from skrough.predict.predict_attrs import predict_attrs
from skrough.structs.attrs_subset import AttrsSubset


def predict_attrs_ensemble(
    model_ensemble: Iterable[AttrsSubset],
    reference_data: np.ndarray,
    reference_data_y: np.ndarray,
    reference_data_y_count: int,
    predict_data: np.ndarray,
    return_proba: bool = False,
    predict_strategy: PredictStrategyKey = "original_order",
    no_answer_strategy: NoAnswerStrategyKey = "nan",
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
        predict_strategy=predict_strategy,
        no_answer_strategy=no_answer_strategy,
        seed=seed,
        n_jobs=n_jobs,
    )
