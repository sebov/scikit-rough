from __future__ import annotations

from typing import Any, Callable, Iterable, Literal, Mapping, get_args

import joblib
import numba
import numba.typed
import numpy as np

import skrough.typing as rght
from skrough.algorithms.meta.processing import RNG_INTEGERS_PARAM
from skrough.dataprep import prepare_factorized_array
from skrough.permutations import get_objs_permutation
from skrough.predict.aggregate import aggregate_predictions
from skrough.structs.group_index import GroupIndex
from skrough.unique import get_uniques_and_positions


@numba.njit
def _predict(
    reference_group_ids: np.ndarray,
    reference_decisions_offsets: np.ndarray,
    reference_decisions: np.ndarray,
    input_group_ids: np.ndarray,
):
    # create group_id -> dec mapping
    group_id_to_dec = {}
    # pylint: disable-next=consider-using-enumerate
    for i in range(len(reference_group_ids)):
        group_id_to_dec[reference_group_ids[i]] = reference_decisions[
            reference_decisions_offsets[i]
        ]

    # create empty result, filled with nan
    result = np.full(len(input_group_ids), fill_value=np.nan, dtype=np.float64)

    # pylint: disable-next=consider-using-enumerate
    for i in range(len(input_group_ids)):
        # check if input_group_id is in the group_id -> dec mapping
        # if so, add the decision to the result
        if input_group_ids[i] in group_id_to_dec:
            result[i] = group_id_to_dec[input_group_ids[i]]
    return result


def predict_strategy_original_order(
    reference_ids: np.ndarray,
    reference_y: np.ndarray,
    predict_ids: np.ndarray,
    seed: rght.Seed = None,  # pylint: disable=unused-argument
) -> Any:
    # prepare unique group_ids and their offsets
    unique_ids, uniques_index = get_uniques_and_positions(reference_ids)

    # prepare the result
    result = _predict(unique_ids, uniques_index, reference_y, predict_ids)

    return result


def predict_strategy_randomized(
    reference_ids: np.ndarray,
    reference_y: np.ndarray,
    predict_ids: np.ndarray,
    seed: rght.Seed = None,
) -> Any:

    reference_permutation = get_objs_permutation(len(reference_ids), seed=seed)
    reference_ids = reference_ids[reference_permutation]
    reference_y = reference_y[reference_permutation]

    result = predict_strategy_original_order(
        reference_ids=reference_ids,
        reference_y=reference_y,
        predict_ids=predict_ids,
        seed=seed,
    )

    return result


PredictStrategy = Literal[
    "original_order",
    "randomized",
]

PREDICT_STRATEGIES: Mapping[PredictStrategy, rght.PredictStrategyFunction] = {
    "original_order": predict_strategy_original_order,
    "randomized": predict_strategy_randomized,
}


class PredictStrategyRunner(rght.PredictStrategyFunction):
    def __init__(self, strategy: PredictStrategy) -> None:
        if strategy not in get_args(PredictStrategy):
            raise ValueError("Unrecognized prediction strategy")
        self.predict_strategy = PREDICT_STRATEGIES[strategy]

    def __call__(
        self,
        reference_ids: np.ndarray,
        reference_y: np.ndarray,
        predict_ids: np.ndarray,
        seed: rght.Seed = None,
    ):
        return self.predict_strategy(
            reference_ids=reference_ids,
            reference_y=reference_y,
            predict_ids=predict_ids,
            seed=seed,
        )


def get_group_ids_reference_and_predict(
    reference_x: np.ndarray,
    predict_x: np.ndarray,
):
    """Get group ids for reference and for predict/input data."""
    data_x = np.row_stack([reference_x, predict_x])
    x, x_counts = prepare_factorized_array(data_x)
    group_index = GroupIndex.from_data(x, x_counts)
    return np.split(group_index.index, [len(reference_x)])


def get_predictions_from_proba(result, counts):
    result = np.where(counts == 0, np.nan, np.argmax(result, axis=1))
    return result


def predict_single(
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    predict_x: np.ndarray,
    predict_strategy: PredictStrategy = "original_order",
    seed: rght.Seed = None,
):
    predict_strategy_runner = PredictStrategyRunner(predict_strategy)

    # pylint: disable-next=unbalanced-tuple-unpacking
    reference_ids, predict_ids = get_group_ids_reference_and_predict(
        reference_x=reference_x,
        predict_x=predict_x,
    )

    result = predict_strategy_runner(
        reference_ids=reference_ids,
        reference_y=reference_y,
        predict_ids=predict_ids,
        seed=seed,
    )

    return result


def predict_ensemble(
    model_predict_fun: Callable,
    model_ensemble: Iterable,
    reference_data: np.ndarray,
    reference_data_y: np.ndarray,
    reference_data_y_count: int,
    predict_data: np.ndarray,
    return_proba: bool = False,
    predict_strategy: PredictStrategy = "original_order",
    seed: rght.Seed = None,
    n_jobs: int | None = None,
):
    rng = np.random.default_rng(seed)
    predictions_collection = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(model_predict_fun)(
            model=model,
            reference_data=reference_data,
            reference_data_y=reference_data_y,
            predict_data=predict_data,
            strategy=predict_strategy,
            seed=rng.integers(RNG_INTEGERS_PARAM),
        )
        for model in model_ensemble
    )

    result, counts = aggregate_predictions(
        n_objs=len(predict_data),
        n_classes=reference_data_y_count,
        predictions_collection=numba.typed.List(predictions_collection),
    )

    if not return_proba:
        result = get_predictions_from_proba(result, counts)

    return result
