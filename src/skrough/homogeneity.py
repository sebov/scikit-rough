import numba
import numpy as np
import numpy.typing as npt


@numba.njit
def get_homogeneity(
    distribution: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """
    Compute decision homogeneity within groups of objects
    """
    if len(distribution.shape) != 2:
        raise ValueError(
            "%s - input distribution should be 2d", get_homogeneity.__name__
        )
    ngroup, ndec = distribution.shape
    result: npt.NDArray[np.int64] = np.ones(ngroup, dtype=np.int64)
    for i in numba.prange(ngroup):
        non_zero_so_far = False
        for j in range(ndec):
            if distribution[i, j] > 0:
                if non_zero_so_far:
                    result[i] = 0
                    break
                non_zero_so_far = True
    return result


HETEROGENEITY_MAX_COLS = 63


@numba.njit
def get_heterogeneity(
    distribution: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """
    Compute decision homogeneity within groups of objects
    """
    if len(distribution.shape) != 2:
        raise ValueError("get_heterogeneity - input distribution should be 2d")
    if distribution.shape[1] > HETEROGENEITY_MAX_COLS:
        raise ValueError("get_heterogeneity - number of columns loo large")
    ngroup, ndec = distribution.shape
    result: npt.NDArray[np.int64] = np.zeros(ngroup, dtype=np.int64)
    for i in numba.prange(ngroup):
        non_zero_values = 0
        heterogeneity_value = np.int64(0)
        for j in range(ndec):
            heterogeneity_value = 2 * heterogeneity_value
            if distribution[i, j] > 0:
                non_zero_values += 1
                heterogeneity_value += 1
        if non_zero_values > 1:
            result[i] = heterogeneity_value
    return result
