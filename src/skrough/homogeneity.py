import numba
import numpy as np
import numpy.typing as npt


@numba.njit
def get_homogeneity(
    distribution: npt.NDArray[np.int64],
) -> npt.NDArray[np.bool_]:
    """
    Compute decision homogeneity within groups of objects
    """
    ngroup, ndec = distribution.shape
    result: npt.NDArray[np.bool_] = np.ones(ngroup, dtype=np.bool_)
    for i in numba.prange(ngroup):
        non_zero_so_far = False
        for j in range(ndec):
            if distribution[i, j] > 0:
                if non_zero_so_far:
                    result[i] = False
                    break
                else:
                    non_zero_so_far = True
    return result
