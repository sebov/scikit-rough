from typing import Any, Literal, Mapping, get_args

import numba
import numpy as np

import skrough.typing as rght
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


def predict(
    model: ObjsAttrsSubset,
    reference_data: np.ndarray,
    reference_data_y: np.ndarray,
    predict_data: np.ndarray,
    strategy: PredictStrategy = "original_order",
    seed: rght.Seed = None,
):
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
