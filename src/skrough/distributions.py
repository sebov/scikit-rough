import numba
import numpy as np


# TODO: group index in numba?
@numba.njit
def get_dec_distribution(
    group_index, n_groups, factorized_dec_values, dec_values_count_distinct
):
    """
    Compute decision distribution within groups of objects
    """
    result = np.zeros((n_groups, dec_values_count_distinct), dtype=np.int_)
    nrow = group_index.shape[0]
    for i in range(nrow):
        result[group_index[i], factorized_dec_values[i]] += 1
    return result
