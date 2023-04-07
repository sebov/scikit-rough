from __future__ import annotations

import numpy as np

import skrough.typing as rght
from skrough.predict.helpers import (
    NoAnswerStrategyKey,
    PredictStrategyKey,
    predict_single,
)
from skrough.structs.attrs_subset import AttrsSubset


def predict_attrs(
    model: AttrsSubset,
    reference_data: np.ndarray,
    reference_data_y: np.ndarray,
    predict_data: np.ndarray,
    predict_strategy: PredictStrategyKey = "original_order",
    no_answer_strategy: NoAnswerStrategyKey = "nan",
    seed: rght.Seed = None,
):
    """Predict actual classes using a single reduct (attrs subset).

    The function predicts actual classes for a model which is a single reduct (or just
    an attrs subset).

    Args:
        model: _description_
        reference_data: _description_
        reference_data_y: _description_
        predict_data: _description_
        strategy: _description_. Defaults to "original_order".
        seed: _description_. Defaults to None.
    Raises:
        ValueError: _description_
    Returns:
        _description_
    """

    return predict_single(
        reference_data=reference_data[:, model.attrs],
        reference_data_y=reference_data_y,
        predict_data=predict_data[:, model.attrs],
        predict_strategy=predict_strategy,
        no_answer_strategy=no_answer_strategy,
        seed=seed,
    )
