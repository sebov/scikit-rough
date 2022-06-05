import numba
import numpy as np
import numpy.typing as npt


@numba.njit
def get_values_distribution(
    groups: npt.NDArray[np.int64],
    groups_count: int,
    values: np.ndarray,
    values_count: int,
) -> npt.NDArray[np.int64]:
    """
    Compute decision distribution within groups of objects
    """
    result = np.zeros((groups_count, values_count), dtype=np.int64)
    nrow = groups.shape[0]
    for i in range(nrow):
        result[groups[i], values[i]] += 1
    return result
