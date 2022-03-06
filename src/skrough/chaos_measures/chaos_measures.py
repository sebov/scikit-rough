import numba
import numpy as np


@numba.njit
def gini_impurity(
    distribution: np.ndarray,
    n_elements: int,
) -> float:
    """
    Compute average gini impurity

    Compute average gini impurity using the following formula

    .. math:: \\sum((1 - \\sum(counts^2)/(\\sum(counts)^2)) * \\sum(counts)) / n

    where counts correspond to rows in distribution
    """
    ngroup, ndec = distribution.shape
    result: float = 0.0
    for i in numba.prange(ngroup):
        group_count = 0
        sum_squared_counts = 0
        for j in range(ndec):
            x = distribution[i, j]
            group_count += x
            sum_squared_counts += x * x
        if group_count > 0:
            result += (1.0 - sum_squared_counts / (group_count * group_count)) * (
                group_count / n_elements
            )
    return result


@numba.njit
def entropy(
    distribution: np.ndarray,
    n_elements: int,
) -> float:
    """
    Compute average entropy
    """
    ngroup, ndec = distribution.shape
    result: float = 0.0
    for i in numba.prange(ngroup):
        group_count = 0
        for j in range(ndec):
            group_count += distribution[i, j]
        if group_count > 0:
            tmp = 0.0
            for j in range(ndec):
                if distribution[i, j] > 0:
                    p = distribution[i, j] / group_count
                    tmp -= p * np.log2(p)
            result += tmp * (group_count / n_elements)
    return result
