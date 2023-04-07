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
    reference_data_y: np.ndarray,
    predict_ids: np.ndarray,
    seed: rght.Seed = None,  # pylint: disable=unused-argument
) -> Any:
    # prepare unique group_ids and their offsets
    unique_ids, uniques_index = get_uniques_and_positions(reference_ids)

    # prepare the result
    result = _predict(unique_ids, uniques_index, reference_data_y, predict_ids)

    return result


def predict_strategy_randomized(
    reference_ids: np.ndarray,
    reference_data_y: np.ndarray,
    predict_ids: np.ndarray,
    seed: rght.Seed = None,
) -> Any:

    reference_permutation = get_objs_permutation(len(reference_ids), seed=seed)
    reference_ids = reference_ids[reference_permutation]
    reference_data_y = reference_data_y[reference_permutation]

    result = predict_strategy_original_order(
        reference_ids=reference_ids,
        reference_data_y=reference_data_y,
        predict_ids=predict_ids,
        seed=seed,
    )

    return result


PredictStrategyKey = Literal[
    "original_order",
    "randomized",
]

PREDICT_STRATEGIES: Mapping[PredictStrategyKey, rght.PredictStrategyFunction] = {
    "original_order": predict_strategy_original_order,
    "randomized": predict_strategy_randomized,
}


class PredictStrategyRunner(rght.PredictStrategyFunction):
    def __init__(self, strategy: PredictStrategyKey) -> None:
        if strategy not in get_args(PredictStrategyKey):
            raise ValueError("Unrecognized prediction strategy")
        self.predict_strategy = PREDICT_STRATEGIES[strategy]

    def __call__(
        self,
        reference_ids: np.ndarray,
        reference_data_y: np.ndarray,
        predict_ids: np.ndarray,
        seed: rght.Seed = None,
    ):
        return self.predict_strategy(
            reference_ids=reference_ids,
            reference_data_y=reference_data_y,
            predict_ids=predict_ids,
            seed=seed,
        )


def no_answer_strategy_nan(
    reference_data_y: np.ndarray,  # pylint: disable=unused-argument
    seed: rght.Seed = None,  # pylint: disable=unused-argument
):
    return np.nan


def no_answer_strategy_most_frequent(
    reference_data_y: np.ndarray,
    seed: rght.Seed = None,
):
    # TODO: implement most-frequent strategy
    raise NotImplementedError


NoAnswerStrategyKey = Literal[
    "nan",
    "most_frequent",
]

NO_ANSWER_STRATEGIES: Mapping[NoAnswerStrategyKey, rght.NoAnswerStrategyFunction] = {
    "nan": no_answer_strategy_nan,
    "most_frequent": no_answer_strategy_most_frequent,
}


class NoAnswerStrategyRunner(rght.NoAnswerStrategyFunction):
    def __init__(self, strategy: NoAnswerStrategyKey) -> None:
        if strategy not in get_args(NoAnswerStrategyKey):
            raise ValueError("Unrecognized no-answer strategy")
        self.no_answer_strategy = NO_ANSWER_STRATEGIES[strategy]

    def __call__(
        self,
        reference_data_y: np.ndarray,
        seed: rght.Seed = None,
    ):
        return self.no_answer_strategy(
            reference_data_y=reference_data_y,
            seed=seed,
        )


def get_group_ids_for_reference_and_predict_data(
    reference_data: np.ndarray,
    predict_data: np.ndarray,
):
    """Get group ids for reference and for predict/input data."""
    data_x = np.row_stack([reference_data, predict_data])
    x, x_counts = prepare_factorized_array(data_x)
    group_index = GroupIndex.from_data(x, x_counts)
    return np.split(group_index.index, [len(reference_data)])


def get_predictions_from_proba(
    predict_proba: np.ndarray,
    counts: np.ndarray,
    no_answer_value=np.nan,
) -> np.ndarray:
    predict_proba = np.where(
        counts == 0, no_answer_value, np.argmax(predict_proba, axis=1)
    )
    return predict_proba


def predict_single(
    reference_data: np.ndarray,
    reference_data_y: np.ndarray,
    predict_data: np.ndarray,
    predict_strategy: PredictStrategyKey,
    no_answer_strategy: NoAnswerStrategyKey,
    seed: rght.Seed = None,
):
    predict_strategy_runner = PredictStrategyRunner(predict_strategy)

    no_answer_strategy_runner = NoAnswerStrategyRunner(no_answer_strategy)

    rng = np.random.default_rng(seed)

    # pylint: disable-next=unbalanced-tuple-unpacking
    reference_ids, predict_ids = get_group_ids_for_reference_and_predict_data(
        reference_data=reference_data,
        predict_data=predict_data,
    )

    result = predict_strategy_runner(
        reference_ids=reference_ids,
        reference_data_y=reference_data_y,
        predict_ids=predict_ids,
        seed=rng.integers(RNG_INTEGERS_PARAM),
    )

    no_answer_value = no_answer_strategy_runner(
        reference_data_y=reference_data, seed=rng.integers(RNG_INTEGERS_PARAM)
    )
    # fix no-answer in-place
    np.nan_to_num(result, copy=False, nan=no_answer_value)

    return result


def predict_ensemble(
    model_predict_fun: Callable,
    model_ensemble: Iterable,
    reference_data: np.ndarray,
    reference_data_y: np.ndarray,
    reference_data_y_count: int,
    predict_data: np.ndarray,
    predict_strategy: PredictStrategyKey,
    no_answer_strategy: NoAnswerStrategyKey,
    return_proba: bool = False,
    seed: rght.Seed = None,
    n_jobs: int | None = None,
):
    no_answer_strategy_runner = NoAnswerStrategyRunner(no_answer_strategy)

    rng = np.random.default_rng(seed)

    predictions_collection = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(model_predict_fun)(
            model=model,
            reference_data=reference_data,
            reference_data_y=reference_data_y,
            predict_data=predict_data,
            predict_strategy=predict_strategy,
            no_answer_strategy="nan",
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
        no_answer_value = no_answer_strategy_runner(
            reference_data_y=reference_data_y, seed=rng.integers(RNG_INTEGERS_PARAM)
        )
        result = get_predictions_from_proba(
            result,
            counts,
            no_answer_value=no_answer_value,
        )

    return result
