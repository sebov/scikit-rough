from __future__ import annotations

from typing import Any, Iterable, Literal, Mapping, get_args

import joblib
import numba
import numba.typed
import numpy as np

import skrough.typing as rght
from skrough.algorithms.meta.processing import RNG_INTEGERS_PARAM
from skrough.dataprep import prepare_factorized_array
from skrough.permutations import get_objs_permutation
from skrough.structs.group_index import GroupIndex
from skrough.structs.objs_attrs_subset import ObjsAttrsSubset
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


def predict_objs_attrs(
    model: ObjsAttrsSubset,
    reference_data: np.ndarray,
    reference_data_y: np.ndarray,
    predict_data: np.ndarray,
    strategy: PredictStrategy = "original_order",
    seed: rght.Seed = None,
):
    """Predict actual classes using a single bireduct (objs+attrs subset).

    The function predicts actual classes for a model which is a single bireduct (or just
    an objs+attrs subset).

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
    if strategy not in get_args(PredictStrategy):
        raise ValueError("Unrecognized prediction strategy")

    # combine reference and input data into one dataset
    reference_x = reference_data[np.ix_(model.objs, model.attrs)]
    predict_x = predict_data[:, model.attrs]
    data_x = np.row_stack([reference_x, predict_x])

    reference_y = reference_data_y[model.objs]

    # get group index for reference and for input
    x, x_counts = prepare_factorized_array(data_x)
    group_index = GroupIndex.from_data(x, x_counts)
    reference_ids = group_index.index[: len(reference_x)]
    predict_ids = group_index.index[len(reference_x) :]  # noqa: E203

    result = PREDICT_STRATEGIES[strategy](
        reference_ids=reference_ids,
        reference_y=reference_y,
        predict_ids=predict_ids,
        seed=seed,
    )

    return result


@numba.njit
def aggregate_predictions(
    n_objs: int, n_classes: int, predictions_collection: numba.typed.List[np.ndarray]
):
    distribution = np.zeros(
        shape=(n_objs, n_classes),
        dtype=np.float64,
    )

    counts = np.zeros(
        shape=n_objs,
        dtype=np.float64,
    )

    for predictions in predictions_collection:
        for i in range(len(predictions)):  # pylint: disable=consider-using-enumerate
            if not np.isnan(predictions[i]):
                counts[i] += 1
                distribution[i, int(predictions[i])] += 1

    for i in range(n_objs):
        if counts[i] == 0:
            distribution[i, :] = np.nan
        else:
            distribution[i, :] /= counts[i]

    return distribution, counts


def predict_objs_attrs_ensemble(
    model: Iterable[ObjsAttrsSubset],
    reference_data: np.ndarray,
    reference_data_y: np.ndarray,
    reference_data_y_count: int,
    predict_data: np.ndarray,
    return_proba: bool = False,
    strategy: PredictStrategy = "original_order",
    seed: rght.Seed = None,
    n_jobs: int | None = None,
):
    if strategy not in get_args(PredictStrategy):
        raise ValueError("Unrecognized prediction strategy")

    rng = np.random.default_rng(seed)
    predictions_collection = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(predict_objs_attrs)(
            model=objs_attrs,
            reference_data=reference_data,
            reference_data_y=reference_data_y,
            predict_data=predict_data,
            strategy=strategy,
            seed=rng.integers(RNG_INTEGERS_PARAM),
        )
        for objs_attrs in model
    )

    result, counts = aggregate_predictions(
        n_objs=len(predict_data),
        n_classes=reference_data_y_count,
        predictions_collection=numba.typed.List(predictions_collection),
    )

    if not return_proba:
        result = np.where(counts == 0, np.nan, np.argmax(result, axis=1))

    return result
