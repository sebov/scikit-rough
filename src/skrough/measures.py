import numba
import numpy as np


def gini_impurity_orig(distribution: np.ndarray, n: np.int_):
    r"""
    Compute average gini impurity

    Compute average gini impurity using the following formula

    .. math:: \sum((1 - \sum(counts^2)/(\sum(counts)^2)) * \sum(counts)) / n

    where counts correspond to rows in distribution

    """
    counts2 = np.sum(distribution ** 2, axis=1)
    group_counts = np.sum(distribution, axis=1)
    # TODO: when the div by zero disaster can happen????
    # divide counts2/group_counts but for "all-0s" rows ignore
    # the "div by zero disaster" and take the result as 1
    # TODO: is this a better solution?
    division = counts2 / group_counts ** 2
    division[~np.isfinite(division)] = 0
    result = np.sum((1 - division) * group_counts) / n
    return result


@numba.njit
def gini_impurity(
    distribution: np.ndarray,
    n_elements: np.int_,
) -> np.float64:
    """
    Compute average gini impurity

    Compute average gini impurity using the following formula

    .. math:: \\sum((1 - \\sum(counts^2)/(\\sum(counts)^2)) * \\sum(counts)) / n

    where counts correspond to rows in distribution
    """
    nrow = distribution.shape[0]
    ncol = distribution.shape[1]
    result = np.float64(0.0)
    for i in numba.prange(nrow):
        group_count = 0
        sum_squared_counts = 0
        for j in range(ncol):
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
    n_elements: np.int64,
) -> np.float64:
    """
    Compute average entropy
    """
    nrow = distribution.shape[0]
    ncol = distribution.shape[1]
    result = np.float64(0.0)
    for i in numba.prange(nrow):
        group_count = 0
        for j in range(ncol):
            group_count += distribution[i, j]
        if group_count > 0:
            tmp = 0.0
            for j in range(ncol):
                if distribution[i, j] > 0:
                    p = distribution[i, j] / group_count
                    tmp -= p * np.log2(p)
            result += tmp * (group_count / n_elements)
    return result
