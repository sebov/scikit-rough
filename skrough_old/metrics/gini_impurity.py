import numpy as np
import numba


def gini_impurity_orig(distribution, n: int):
    """
    Compute average gini impurity

    Compute average gini impurity using the following formula

    .. math:: \sum((1 - \sum(counts^2)/(\sum(counts)^2)) * \sum(counts)) / n

    where counts correspond to rows in distribution

    """
    counts2 = np.sum(distribution ** 2, axis=1)
    group_counts = np.sum(distribution, axis=1)
    # TODO: when the div by zero disaster can happen????
    # divide counts2/group_counts but for "all-0s" rows ignore the "div by zero disaster"
    # and take the result as 1
    # TODO: is this a better solution?
    division = counts2 / group_counts ** 2
    division[~np.isfinite(division)] = 0
    result = np.sum((1 - division) * group_counts) / n
    return result


@numba.njit(parallel=False, fastmath=False)
def gini_impurity(distribution: np.array, n: int):
    nrow = distribution.shape[0]
    ncol = distribution.shape[1]
    counts2 = np.zeros(nrow, dtype=np.int_)
    group_counts = np.zeros(nrow, dtype=np.int_)
    for i in numba.prange(nrow):
        for j in range(ncol):
            x = distribution[i, j]
            counts2[i] += x * x
            group_counts[i] += x
    result = 0.0
    for i in numba.prange(nrow):
        if group_counts[i] > 0:
            result += (1.0 - counts2[i] / (group_counts[i] * group_counts[i])) * (group_counts[i] / n)
    return result
