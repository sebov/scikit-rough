"""Rough sets related functions."""

from typing import List, Tuple

import numba
import numpy as np

import skrough.typing as rght
from skrough.homogeneity import encode_homogeneity
from skrough.structs.group_index import GroupIndex
from skrough.utils import get_positions_where_values_in


def get_positive_region(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    attrs: rght.IndexListLike,
) -> List[int]:
    """Compute the positive region with respect to given attributes.

    The positive region :math:`POS_B(d)` consists of all objects that can be
    uniquely classified to decision classes using attributes in :math:`B`. It is
    the union of those equivalence classes (induced by ``attrs``) within which
    all objects share the same decision value.

    Args:
        x: Factorized data table representing conditional attributes. Values should be 0-based
            integer indices.
        x_counts: Number of distinct values for each conditional attribute.
        y: Factorized decision values for the objects. Values should be 0-based integer indices.
        y_count: Number of distinct decision attribute values.
        attrs: Subset of conditional attributes to use. Given as a sequence of column indices,
            or :obj:`None` to use all attributes.

    Returns:
        List of object positions (0-based indices into ``x``) that belong to the positive region.

    Examples:
        >>> x = np.asarray([[0, 1], [0, 1], [1, 0], [1, 0], [2, 2], [2, 2]])
        >>> x_counts = np.asarray([3, 3])
        >>> y = np.asarray([0, 0, 1, 1, 0, 1])
        >>> y_count = 2
        >>> get_positive_region(x, x_counts, y, y_count, attrs=[0, 1])
        [0, 1, 2, 3]
    """
    group_index = GroupIndex.from_data(x, x_counts, attrs)
    dec_distribution = group_index.get_distribution(y, y_count)
    homogeneity = encode_homogeneity(dec_distribution)
    # compute positions in ``homogeneity`` (here positions correspond to group ids) that
    # are equal to 1 (homogeneous groups)
    homogeneous_groups = homogeneity.nonzero()[0]
    # return positions in group_index (they correspond to objects) for which values
    # belong to the set of homogeneous groups
    return get_positions_where_values_in(group_index.index, homogeneous_groups)


def get_gamma_value(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    attrs: rght.IndexListLike,
) -> float:
    """Compute the dependency degree between attributes and decision.

    The dependency degree :math:`\\gamma(B)` measures how well the attribute subset :math:`B`
    determines the decision attribute. It is defined as the ratio of the positive region size to the
    universe size:

    .. math::

      \\gamma(B) = \\frac{|POS_B(d)|}{|U|}

    A value of ``1`` means the decision table is consistent with respect to ``attrs`` (all objects
    can be uniquely classified). A value of ``0`` means no object can be uniquely classified.

    Args:
        x: Factorized data table representing conditional attributes. Values should be 0-based
            integer indices.
        x_counts: Number of distinct values for each conditional attribute.
        y: Factorized decision values for the objects. Values should be 0-based integer indices.
        y_count: Number of distinct decision attribute values.
        attrs: Subset of conditional attributes to use. Given as a sequence of column indices, or
            :obj:`None` to use all attributes.

    Returns:
        Dependency degree in the range :math:`[0, 1]`. Returns ``1`` for empty input.

    Examples:
        >>> x = np.asarray([[0, 1], [0, 1], [1, 0], [1, 0], [2, 2], [2, 2]])
        >>> x_counts = np.asarray([3, 3])
        >>> y = np.asarray([0, 0, 1, 1, 0, 1])
        >>> y_count = 2
        >>> round(get_gamma_value(x, x_counts, y, y_count, attrs=[0, 1]), 4)
        0.6667
        >>> y_consistent = np.asarray([0, 0, 1, 1, 2, 2])
        >>> get_gamma_value(x, x_counts, y_consistent, 3, attrs=[0, 1])
        1.0
    """
    if len(x) == 0:
        return 1
    pos = get_positive_region(x, x_counts, y, y_count, attrs)
    return len(pos) / len(x)


@numba.njit(cache=True)
def get_lower_upper_group_ids(
    membership_distr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute group ids belonging to lower and upper approximations.

    Given a membership distribution with two columns (column 0 = count of objects not in the target
    set, column 1 = count of objects in the target set), this function identifies which groups
    belong to the lower and upper approximations of the target set.

    A group belongs to the **upper approximation** if it has a non-empty intersection with the
    target set (at least one object from the target set).

    A group belongs to the **lower approximation** if it is fully contained within the target set
    (no objects outside the target set).

    Args:
        membership_distr: A 2D array with shape ``(n_groups, 2)`` where each row
            represents a group and the two columns represent counts of objects outside (column 0)
            and inside (column 1) the target set.

    Returns:
        A tuple of two arrays:

        - Array of group ids belonging to the lower approximation.
        - Array of group ids belonging to the upper approximation.

    Raises:
        ValueError: If ``membership_distr`` is not a 2D array with exactly two columns.

    Examples:
        >>> get_lower_upper_group_ids(np.asarray([[0, 3], [2, 1], [4, 0], [0, 0]]))
        (array([0]), array([0, 1]))
    """
    if membership_distr.ndim != 2 or membership_distr.shape[1] != 2:
        raise ValueError(
            "Membership distribution should be a 2D array of just two columns"
        )
    lower = []
    upper = []
    ngroup = len(membership_distr)
    for i in numba.prange(ngroup):  # pylint: disable=not-an-iterable
        if membership_distr[i, 1] > 0:
            upper.append(i)
            if membership_distr[i, 0] == 0:
                lower.append(i)
    return np.asarray(lower), np.asarray(upper)


def get_approximations(
    x: np.ndarray,
    x_counts: np.ndarray,
    attrs: rght.IndexListLike,
    concept: rght.IndexListLike,
) -> Tuple[List[int], List[int]]:
    """Compute lower and upper approximations of a concept.

    Given a concept (a subset of objects) and a set of attributes, this function computes the lower
    and upper approximations of the concept with respect to the equivalence classes induced by
    ``attrs``.

    The **lower approximation** consists of all objects whose equivalence class is fully contained
    within the concept.

    The **upper approximation** consists of all objects whose equivalence class has a non-empty
    intersection with the concept.

    Args:
        x: Factorized data table representing conditional attributes. Values should be 0-based
            integer indices.
        x_counts: Number of distinct values for each conditional attribute.
        attrs: Subset of conditional attributes to use. Given as a sequence of column indices, or
            :obj:`None` to use all attributes.
        concept: A subset of object positions (0-based indices into ``x``) defining the concept to
            approximate.

    Returns:
        A tuple of two lists:

        - List of object positions belonging to the lower approximation.
        - List of object positions belonging to the upper approximation.

    Examples:
        >>> x = np.asarray([[0, 1], [0, 1], [1, 0], [1, 0], [2, 2], [2, 2]])
        >>> x_counts = np.asarray([3, 3])
        >>> concept = [0, 1, 2]
        >>> get_approximations(x, x_counts, attrs=[0, 1], concept=concept)
        ([0, 1], [0, 1, 2, 3])
    """
    group_index = GroupIndex.from_data(x, x_counts, attrs)
    # treat membership as a decision attribute for this computation
    # imposed interpretation: 0 - not in concept, 1 - in concept
    membership = np.isin(np.arange(len(x)), concept).astype(int)
    membership_count = 2
    membership_distr = group_index.get_distribution(membership, membership_count)
    lower_group_ids, upper_group_ids = get_lower_upper_group_ids(membership_distr)
    lower = get_positions_where_values_in(group_index.index, lower_group_ids)
    upper = get_positions_where_values_in(group_index.index, upper_group_ids)
    return lower, upper
