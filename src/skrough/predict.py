import numba
import numpy as np

from skrough.dataprep import prepare_factorized_x
from skrough.structs.group_index import GroupIndex
from skrough.structs.objs_attrs_subset import ObjsAttrsSubset


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


def predict(
    model: ObjsAttrsSubset,
    reference_data: np.ndarray,
    reference_data_dec: np.ndarray,
    predict_data: np.ndarray,
) -> np.ndarray:
    # combine reference and input data into one dataset
    reference_x = reference_data[np.ix_(model.objs, model.attrs)]
    predict_x = predict_data[:, model.attrs]
    reference_y = reference_data_dec[model.objs]
    data_x = np.row_stack([reference_x, predict_x])

    # get group index for reference and for input
    x, x_counts = prepare_factorized_x(data_x)
    group_index = GroupIndex.create_from_data(x, x_counts, range(x.shape[1]))
    group_index_reference = group_index.index[: len(reference_x)]
    group_index_input = group_index.index[len(reference_x) :]  # noqa: E203

    # prepare unique group_ids and their offsets
    unique_ids, unique_index = np.unique(group_index_reference, return_index=True)

    # prepare the result
    result = _predict(unique_ids, unique_index, reference_y, group_index_input)

    return result
