import numpy as np
import numpy.typing as npt


def get_homogeneity(
    distribution: npt.NDArray[np.int64],
) -> npt.NDArray[np.bool_]:
    """
    Compute decision homogeneity within groups of objects
    """
    # check in which rows there are no more than one positive values
    return np.sum(distribution > 0, axis=1) <= 1
