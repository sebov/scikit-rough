# pylint: disable=duplicate-code

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

import skrough.typing as rght
from skrough.predict.helpers import (
    NoAnswerStrategyKey,
    PredictionResultPreparer,
    PredictStrategyKey,
    check_reference_data,
    predict_ensemble,
)
from skrough.predict.predict_attrs import predict_attrs
from skrough.structs.attrs_subset import AttrsSubset


def predict_attrs_ensemble(
    model_ensemble: Iterable[AttrsSubset],
    reference_data: np.ndarray,
    reference_data_y: np.ndarray,
    predict_data: np.ndarray,
    return_proba: bool = False,
    predict_strategy: PredictStrategyKey = "majority",
    no_answer_strategy: NoAnswerStrategyKey = "missing",
    raw_mode: bool = False,
    fill_missing: Any = np.nan,
    preferred_prediction_dtype: type[np.generic] | None = None,
    seed: rght.Seed = None,
    n_jobs: int | None = None,
):
    # TODO: add to docstring that if no_answer_strategy is "missing" but
    # missing_decision is set to some "X" (assuming "X" being an actual decision
    # available) then prediction result may contain "X" as the answer but predict_proba
    # will have a row with all nans therefore this may lead to inconsistency (in such
    # the case) between predictions and proba

    check_reference_data(
        reference_data=reference_data, reference_data_y=reference_data_y
    )

    result_preparer = PredictionResultPreparer.from_reference_data_y(
        reference_data_y=reference_data_y,
        raw_mode=raw_mode,
        fill_missing=fill_missing,
        preferred_prediction_dtype=preferred_prediction_dtype,
    )

    result = predict_ensemble(
        model_predict_fun=predict_attrs,
        model_ensemble=model_ensemble,
        reference_data=reference_data,
        reference_data_y=result_preparer.y,
        reference_data_y_count=len(result_preparer.y_uniques),
        predict_data=predict_data,
        return_proba=return_proba,
        predict_strategy=predict_strategy,
        no_answer_strategy=no_answer_strategy,
        seed=seed,
        n_jobs=n_jobs,
    )

    if not return_proba:
        return result_preparer.prepare(result)

    return result, result_preparer.y_uniques
