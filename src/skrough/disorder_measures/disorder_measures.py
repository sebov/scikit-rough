import numba
import numpy as np


@numba.njit(cache=True)
def gini_impurity(
    distribution: np.ndarray,
    n_elements: int,
) -> float:
    """Compute average Gini impurity.

    Compute average Gini impurity for a given input distribution. Gini impurity measure is commonly
    used in decision tree algorithms to make decisions about node splits. It expresses the
    likelihood of a random sample being misclassified if it is given a random label according to the
    class distribution in the dataset.

    Gini impurity is computed using the following formula:

    .. math::
        \\sum_{i}
            \\frac{\\sum_{j} distribution_{ij}}{n\\_elements}
            \\cdot
            \\left(1 - \\frac{\\sum_{j} distribution_{ij}^2}{(\\sum_{j} distribution_{ij})^2}\\right)

    Despite the actual semantic of Gini impurity measure, the function is commonly used to express
    the amount of disorder or uncertainty in a single group of observations (or averaged over a
    collection of groups of observations).

    The distribution format is defined as a 2D array where:

    - rows correspond to separate contexts, e.g., groups of objects or equivalence classes,
    - values in columns for a particular row represent discrete distribution, i.e., the number
      of occurrences of each possible decision attribute distinct value.

    Args:
        distribution: A 2D array representing a distribution.
        n_elements: Number of elements represented by the input distribution. It is given
            explicitly to avoid recomputing it from the distribution.

    Returns:
        Average Gini impurity for the given input distribution.

    Examples:
        >>> gini_impurity(np.asarray([[2, 0], [1, 1], [0, 3]]), 7)
        0.14285714285714285
        >>> gini_impurity(np.asarray([[1, 1], [1, 1]]), 4)
        0.5
        >>> gini_impurity(np.asarray([[5, 0], [0, 3]]), 8)
        0.0
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


@numba.njit(cache=True, fastmath=True)
def entropy(
    distribution: np.ndarray,
    n_elements: int,
) -> float:
    """Compute average entropy.

    Compute average entropy (using base-2 logarithm) for a given input distribution. Entropy is a
    measure of uncertainty or disorder in a dataset, commonly used in decision tree algorithms and
    information theory.

    Entropy is computed using the following formula:

    .. math::
        \\sum_{i}
            \\frac{\\sum_{j} distribution_{ij}}{n\\_elements}
            \\cdot
            \\left(-\\sum_{j} \\frac{distribution_{ij}}{\\sum_{k} distribution_{ik}}
            \\cdot \\log_2\\left(\\frac{distribution_{ij}}{\\sum_{k} distribution_{ik}}\\right)\\right)

    The distribution format is defined as a 2D array where:

    - rows correspond to separate contexts, e.g., groups of objects or equivalence classes,
    - values in columns for a particular row represent discrete distribution, i.e., the number
      of occurrences of each possible decision attribute distinct value.

    Args:
        distribution: A 2D array representing a distribution.
        n_elements: Number of elements represented by the input distribution. It is given
            explicitly to avoid recomputing it from the distribution.

    Returns:
        Average entropy for the given input distribution.

    Examples:
        >>> entropy(np.asarray([[2, 0], [1, 1], [0, 3]]), 7)
        0.2857142857142857
        >>> entropy(np.asarray([[1, 1], [1, 1]]), 4)
        1.0
        >>> entropy(np.asarray([[5, 0], [0, 3]]), 8)
        0.0
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


@numba.njit(cache=True)
def conflicts_count(
    distribution: np.ndarray,
    n_elements: int,  # pylint: disable=unused-argument
) -> float:
    """Compute the total number of conflicting pairs.

    Compute the number of pairs of objects that belong to the same group but have different
    decision values. A conflict occurs when two objects are indiscernible with respect to
    conditional attributes (i.e., they are in the same equivalence class) but have different
    decision attribute values.

    The conflicts count is computed using the following formula:

    .. math::
        \\frac{\\sum_{i} \\left((\\sum_{j} distribution_{ij})^2 - \\sum_{j} distribution_{ij}^2\\right)}{2}

    The distribution format is defined as a 2D array where:

    - rows correspond to separate contexts, e.g., groups of objects or equivalence classes,
    - values in columns for a particular row represent discrete distribution, i.e., the number
      of occurrences of each possible decision attribute distinct value.

    Args:
        distribution: A 2D array representing a distribution.
        n_elements: Number of elements represented by the input distribution. This parameter
            is unused but kept for interface consistency with other disorder measures.

    Returns:
        Total number of conflicting pairs across all groups.

    Examples:
        >>> conflicts_count(np.asarray([[2, 0], [1, 1], [0, 3]]), 7)
        1.0
        >>> conflicts_count(np.asarray([[1, 1], [1, 1]]), 4)
        2.0
        >>> conflicts_count(np.asarray([[5, 0], [0, 3]]), 8)
        0.0
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
