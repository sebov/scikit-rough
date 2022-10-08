"""Functions related to homogeneity/heterogeneity of decision tables."""

from typing import Tuple

import numba
import numpy as np
import numpy.typing as npt

import skrough.typing as rght
from skrough.structs.group_index import GroupIndex
from skrough.unique import get_uniques_and_compacted


@numba.njit
def get_homogeneity(
    distribution: npt.NDArray[np.int64],
) -> npt.NDArray[np.int8]:
    """Compute distribution homogeneity.

    Compute homogeneity for a given input distribution. The function is mainly used for
    computation of homogeneity of decision attributes. The distribution format is
    defined as a 2D array where:

    - rows correspond to separate contexts, e.g., groups of objects or equivalence
      classes,
    - values in columns for a particular row represent discrete distribution, i.e.,
      the number of occurrences of each possible decision attribute distinct value.

    The result is a sequence of integer values (``0`` or ``1``), where each corresponds
    to a group/context (row) in the ``distribution`` input. A value of ``1`` means that
    there is at most one non-zero value in a given row (meaning that a row is
    homogenous), ``0`` otherwise (non-homogenous).

    Args:
        distribution: A 2D array representing a distribution.

    Raises:
        ValueError: If ``distribution`` is not a two-dimensional array.

    Returns:
        An array consisting of integer values ``0`` or ``1`` indicating that a
        corresponding row in the ``distribution`` input argument is either
        non-homogenous (for ``0``) or homogenous (for ``1``).

    Examples:
        >>> get_homogeneity(
        ...     np.asarray(
        ...         [
        ...             [0, 0],
        ...             [1, 1],
        ...             [0, 3],
        ...             [5, 0],
        ...         ]
        ...     )
        ... )
        array([1, 0, 1, 1])
    """
    if distribution.ndim != 2:
        raise ValueError("input `distribution` should be 2D")
    ngroup, ndec = distribution.shape
    result: npt.NDArray[np.int8] = np.ones(ngroup, dtype=np.int8)
    for i in numba.prange(ngroup):  # pylint: disable=not-an-iterable
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
    """Compute distribution heterogeneity.

    Compute heterogeneity for a given input distribution. The function is mainly used
    for computation of heterogeneity of decision attributes. The distribution format is
    defined as a 2D array where:

    - rows correspond to separate contexts, e.g., groups of objects or equivalence
      classes,
    - values in columns for a particular row represent discrete distribution, i.e.,
      the number of occurrences of each possible decision attribute distinct value.

    The result is a sequence of integer values (``0`` or :code:`>=1`), where each
    corresponds to a group/context (row) in the ``distribution`` input. A value of ``0``
    means that there is at most one non-zero value in a given row (meaning that a row is
    non-heterogenous/homogenous). Values :code:`>=1` represent heterogenous rows, where
    different positive values show different kinds of heterogeneity. E.g., the function
    distinguishes a row where there are non zero values on positions ``0`` and ``1``
    from a row where there are non zero values on positions ``1`` and ``2``. The actual
    return value :code:`>=1` that corresponds to a given row is created as a binary
    represented number with bits set for places where discrete distribution counts are
    greater than ``0``.

    Args:
        distribution: A 2D array representing a distribution.

    Raises:
        ValueError: If ``distribution`` is not a two-dimensional array.
        ValueError: If the number of columns in the ``distribution`` input argument is
            greater than ``63``.

    Returns:
        An array consisting of integer values ``0`` or :code:`>=1` indicating that a
        corresponding row in the ``distribution`` input argument is either
        non-heterogenous/homogenous (for ``0``) or heterogenous (for :code:`>=1`).

    Examples:
        >>> get_heterogeneity(
        ...     np.asarray(
        ...         [
        ...             [0, 0, 0],
        ...             [1, 0, 0],
        ...             [0, 1, 0],
        ...             [0, 0, 1],
        ...             [1, 1, 0],
        ...             [1, 9, 0],
        ...             [9, 1, 0],
        ...             [1, 0, 1],
        ...             [1, 0, 9],
        ...             [9, 0, 1],
        ...             [0, 1, 1],
        ...             [0, 9, 1],
        ...             [0, 1, 9],
        ...             [1, 1, 1],
        ...             [1, 8, 9],
        ...             [8, 9, 1],
        ...         ]
        ...     )
        ... )
        array([0, 0, 0, 0, 6, 6, 6, 5, 5, 5, 3, 3, 3, 7, 7, 7])
    """
    if distribution.ndim != 2:
        raise ValueError("input `distribution` should be 2D")
    if distribution.shape[1] > HETEROGENEITY_MAX_COLS:
        raise ValueError("number of columns in `distribution` is too large")
    ngroup, ndec = distribution.shape
    result: npt.NDArray[np.int64] = np.zeros(ngroup, dtype=np.int64)
    for i in numba.prange(ngroup):  # pylint: disable=not-an-iterable
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


@numba.njit
def _groups_decisions_replace(
    group_index: np.ndarray,
    y: np.ndarray,
    y_count: int,
    replace: np.ndarray,
) -> np.ndarray:
    """Replace decisions in groups.

    Replace decisions in groups according to the given ``group_decisions``. The function
    interprets ``replace`` argument as a mapping, where:

    - positions represent group ids and they are used as keys in the mapping
    - values represent new decision values and they are used as values in the mapping

    So, effectively, it maps :code:`group ids -> new decisions`. When decision value in
    the mapping equals to ``0`` then it has a special handling and it is interpreted as
    an instruction to preserve the original decision for an object.

    Args:
        group_index: Sequence of group ids that represents split of the objects
            represented by this structure into groups.
        y: Factorized decision values for the objects represented by the input
            ``group_index``. The values should be given in a form of integer-location
            based indexing sequence of the factorized decision values, i.e., 0-based
            values that index distinct decisions.
        y_count: Number of distinct decision attribute values.
        group_decisions: A mapping of objects groups to decisions, given as a sequence
            of decision ids where positions in the sequence represent group ids. The
            mapping represented in this way is used to change original object decisions
            to new decisions encoded in the mapping. The following rules are applied:

            - if a given object has a group (in terms of the ``group_index`` input) that
              maps to ``0`` (in terms of the ``group_decisions`` mapping) then the
              original object's decision is preserved
            - otherwise, a given object is assigned a new decision using the following
              expression::

                y_count - 1 + replace[group_index[i]]

              i.e., a new decision value that is greater than the original range of
              possible values (``y_count``) is assigned according to the given
              ``replace`` argument

    Returns:
        New decision values created from the input ``y`` changed according to the
        ``group_decisions`` values.
    """
    result = np.empty_like(y)
    for i in numba.prange(len(y)):  # pylint: disable=not-an-iterable
        if replace[group_index[i]] == 0:
            # ``0`` is reserved for non-heterogenous groups, so we preserve the original
            # decision
            result[i] = y[i]
        else:
            # values > ``0`` represents heterogenous groups, so we set new decisions for
            # objects belonging to those groups based on ``heterogeneity_values``; but
            # the new decision values need to be numbered accordingly, i.e., the values
            # need to be shifted behind the original range of decisions
            # ``0..(y_count-1)``
            result[i] = y_count - 1 + replace[group_index[i]]
    return result


def heterogeneous_groups_decisions_replace(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    attrs: rght.LocationsLike,
    distinguish_generalized_decisions: bool = False,
) -> Tuple[np.ndarray, int]:
    """Return consistent decision values.

    Prepare new decision values in a way that makes data consistent (in the meaning of a
    consistent decision table). The groups (equivalence classes in the context of the
    indiscernibility relation) are induced from the given dataset ``x`` and a subset of
    attributes ``attrs``. Original decisions ``y`` are then processed to prepare new
    consistent decision values. It is done by preserving decision values for homogenous
    groups and replacing decisions for objects from heterogenous ones. The
    ``distinguish_generalized_decisions`` boolean flag can be used to control whether
    heterogenous groups should be distinguished from each other
    (:code:`distinguish_generalized_decisions is True`) or treated equally
    (:code:`distinguish_generalized_decisions is False`). Distinguishing the
    heterogenous groups means that objects from groups of different characteristics (a
    different subset of decision values appearing in a group, cf.
    :func:`~skrough.homogeneity.get_heterogeneity`) are assigned different new decision
    values. When heterogenous groups are not to be distinguished then objects from all
    heterogenous groups are assigned the same new decision value.

    Args:
        x: Factorized data table representing conditional features/attributes for the
            objects the computation should be performed on. The values in each column
            should be given in a form of integer-location based indexing sequence of the
            factorized conditional attribute values, i.e., 0-based values that index
            distinct values of the conditional attribute.
        x_counts: Number of distinct attribute values given for each conditional
            attribute. The argument is expected to be given as a 1D array.
        y: Factorized decision values for the objects represented by the input
            :obj:`x` argument. The values should be given in a form of integer-location
            based indexing sequence of the factorized decision values, i.e., 0-based
            values that index distinct decisions.
        y_count: Number of distinct decision attribute values.
        attrs: A subset of conditional attributes the check should be performed on.
            It should be given in a form of a sequence of integer-location based
            indexing of the selected conditional attributes from ``x``. :obj:`None`
            value means to use all available conditional attributes. Defaults to
            :obj:`None`.
        distinguish_generalized_decisions: A flag to control whether heterogenous groups
            should be distinguished from each other or not. Defaults to :obj:`False`.

    Returns:
        New decision values returned in a form of 2-element tuple with the following
        elements

        - factorized decision attribute returned in form of 1d array
        - decision attribute domain size

        The new decision values together with the input data ``x`` and ``x_counts`` form
        a consistent decision table.

    Examples:
        >>> from skrough.dataprep import (
        ...     prepare_factorized_array,
        ...     prepare_factorized_vector
        ... )
        >>> x, x_counts = prepare_factorized_array(np.asarray([[8, 8, 8],
        ...                                                    [8, 8, 8],
        ...                                                    [1, 7, 8],
        ...                                                    [1, 8, 8],
        ...                                                    [1, 1, 8],
        ...                                                    [1, 1, 1]]))
        >>> y, y_count = prepare_factorized_vector(np.asarray([3, 4, 8, 9, 4, 5]))
        >>> y, y_count
        (array([0, 1, 2, 3, 1, 3]), 5)
        >>> replace_heterogeneous_groups_decisions(
        ...     x,
        ...     x_counts,
        ...     y,
        ...     y_count,
        ...     attrs=[0, 1],
        ...     distinguish_generalized_decisions=False,
        ... )
        (array([5, 5, 2, 3, 5, 5]), 6)
        >>> replace_heterogeneous_groups_decisions(
        ...     x,
        ...     x_counts,
        ...     y,
        ...     y_count,
        ...     attrs=[0, 1],
        ...     distinguish_generalized_decisions=True,
        ... )
        (array([6, 6, 2, 3, 5, 5]), 7)
    """
    if len(x) == 0:
        return y, y_count

    group_index = GroupIndex.from_data(x, x_counts, attrs)
    dec_distribution = group_index.get_distribution(y, y_count)
    if distinguish_generalized_decisions:
        heterogeneity = get_heterogeneity(dec_distribution)
    else:
        heterogeneity = 1 - get_homogeneity(dec_distribution)

    # values ``0`` (if present) mean non-heterogenous groups, i.e., homogenous groups
    # values > ``0`` (if present) mean heterogenous groups
    # get_uniques_and_compacted returns unique elements as sorted (ascending) sequence
    heterogeneity_unique_values, heterogeneity_compacted = get_uniques_and_compacted(
        heterogeneity
    )
    # let's compute the number of heterogenous groups
    heterogenous_groups_count = len(heterogeneity_unique_values)
    if heterogeneity_unique_values[0] == 0:
        # if the value ``0`` is there, we need to adjust, i.e.,
        # decrease heterogenous_group_count by 1
        heterogenous_groups_count -= 1
    else:
        # otherwise, we need to adjust ``heterogeneity_compacted``, as ``0`` now
        # represents actual heterogenous group but we want to keep ``0`` reserved for
        # non-heterogenous ones
        heterogeneity_compacted += 1

    result = _groups_decisions_replace(
        group_index=group_index.index,
        y=y,
        y_count=y_count,
        replace=heterogeneity_compacted,
    )

    return result, (y_count + heterogenous_groups_count)
