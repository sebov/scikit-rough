import numba
import numpy as np


@numba.njit
def gini_impurity(
    distribution: np.ndarray,
    n_elements: int,
) -> float:
    """
    Compute average gini impurity.

    Compute average gini impurity for a given input distribution. Gini impurity measure
    is commonly used in decision tree algorithms to make decisions about node splits. It
    expresses the likelihood of a random sample being misclassified if it is given a
    random label according to the class distribution in the dataset.

    Gini impurity is computed using the following formula:

    .. math::
        \\sum(
            (\\frac{\\sum(counts)}{n})
            *
            (1 - \\frac{\\sum(counts^2)}{\\sum(counts)^2})
        )

    where counts correspond to rows in distribution

    Despite the actual semantic of gini impurity measure, the function is commonly used
    to express the amount of disorder, chaos or uncertainty in a single group of
    observations (or averaged over a collection of groups of observations).

    The distribution format is defined as a 2D array where:

    - rows correspond to separate contexts, e.g., groups of objects or equivalence
      classes,
    - values in columns for a particular row represent discrete distribution, i.e.,
      the number of occurrences of each possible decision attribute distinct value.



    Args:
        distribution: A 2D array representing a distribution.
        n_elements: Number of elements represented by the input distribution. It is
            given

    Returns:
        Average gini impurity for the given input distribution.

    """
    ngroup, ndec = distribution.shape
    result: float = 0.0
    for i in numba.prange(ngroup):  # pylint: disable=not-an-iterable
        group_count = 0
        sum_squared_counts = 0
        for j in range(ndec):
            x = distribution[i, j]
            group_count += x
            sum_squared_counts += x * x
        if group_count > 0:
            group_gini = 1.0 - sum_squared_counts / (group_count * group_count)
            group_weight = group_count / n_elements
            result += group_gini * group_weight
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
    for i in numba.prange(ngroup):  # pylint: disable=not-an-iterable
        group_count = 0
        for j in range(ndec):
            group_count += distribution[i, j]
        if group_count > 0:
            tmp = 0.0
            for j in range(ndec):
                if distribution[i, j] > 0:
                    prob = distribution[i, j] / group_count
                    tmp -= prob * np.log2(prob)
            result += tmp * (group_count / n_elements)
    return result


@numba.njit
def conflicts_count(
    distribution: np.ndarray,
    n_elements: int,  # pylint: disable=unused-argument
) -> float:
    """_summary_

    _extended_summary_

    Args:
        distribution: _description_
        n_elements: _description_

    Returns:
        _description_
    """
    ngroup, ndec = distribution.shape
    result: int = 0
    for i in numba.prange(ngroup):  # pylint: disable=not-an-iterable
        group_count = 0
        sum_squared_counts = 0
        for j in range(ndec):
            x = distribution[i, j]
            group_count += x
            sum_squared_counts += x * x
        result += group_count * group_count - sum_squared_counts
    return result / 2
