import numba
import numpy as np
import numpy.typing as npt


@numba.njit
def get_values_distribution(
    group_index_index: npt.NDArray[np.int64],
    group_index_count: int,
    factorized_dec_values: np.ndarray,
    dec_values_count_distinct: int,
) -> npt.NDArray[np.int64]:
    """
    Compute decision distribution within groups of objects
    """
    result = np.zeros((group_index_count, dec_values_count_distinct), dtype=np.int64)
    nrow = group_index_index.shape[0]
    for i in range(nrow):
        result[group_index_index[i], factorized_dec_values[i]] += 1
    return result
