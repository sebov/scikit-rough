from typing import Any, Optional, Sequence

import numpy as np

from skrough.group_index import batch_split_into_groups
from skrough.instances import choose_objects


def get_nunique_objs(x: np.ndarray) -> int:
    """Compute the number of unique rows.

    Compute the number of unique rows. Degenerated tables are handled accordingly,
    i.e., a table with no columns has 1 unique rows if only it has at least one row,
    otherwise it is 0.

    Args:
        x: Input data table.

    Returns:
        Number of unique rows.
    """
    return np.unique(x, axis=0).shape[0]


def check_if_functional_dependency(
    x: np.ndarray,
    y: np.ndarray,
    objs: Optional[Sequence[int]] = None,
    attrs: Optional[Sequence[int]] = None,
) -> bool:
    """Check functional dependency between conditional attributes and the decision.

    Check functional dependency between conditional attributes and the decision. The
    check is based on the number of duplicated rows induced by the given subset of
    attributes either with and without the decision attribute. If the number of
    duplicated rows is the same, the functional dependency holds. The check can be
    further narrowed to the given subset of attributes and objects.

    Args:
        x: Input data table.
        y: Input decision.
        objs: A subset of object that the check should be performed on. It should
            be given in a form of a sequence of integer-location based indexing of the
            selected objects/rows/instances from ``x``. ``None`` value means to use
            all available objects. Defaults to None.
        attrs: A subset of conditional attributes the check should be performed on.
            It should be given in a form of a sequence of integer-location based
            indexing of the selected conditional attributes from ``x``. ``None`` value
            means to use all available conditional attributes. Defaults to None.

    Returns:
        Indication whether functional dependency holds for the given input.
    """
    objects = objs if objs is not None else slice(None)
    attributes = attrs if attrs is not None else slice(None)
    xx_index_expr: Any
    SEQ = (Sequence, np.ndarray)
    if isinstance(objects, SEQ) and isinstance(attributes, SEQ):
        # we want to take all ``objects`` x ``attributes``
        xx_index_expr = np.ix_(objects, attributes)
    else:
        xx_index_expr = np.index_exp[objects, attributes]
    xx = x[xx_index_expr]
    yy = y[objects]
    xxyy = np.hstack((xx, np.expand_dims(yy, axis=1)))
    nunique = get_nunique_objs(xx)
    nunique_with_dec = get_nunique_objs(xxyy)
    return nunique == nunique_with_dec


def check_if_consistent_table(
    x: np.ndarray,
    y: np.ndarray,
) -> bool:
    """Check if decision table is consistent.

    Check if decision table is consistent, i.e., check if it is possible to discern
    objects with different decisions by means of conditional attributes. It is realized
    just as a simple wrapper around ``check_if_functional_dependency`` function
    using all available objects and attributes.

    Args:
        x: Input data table.
        y: Input decision.

    Returns:
        Indication whether the decision table is consistent.
    """
    return check_if_functional_dependency(x, y)


def check_if_reduct(
    x: np.ndarray,
    y: np.ndarray,
    attrs: list[int],
    consistent_table_check: bool = True,
) -> bool:
    """Check if specified attributes form a reduct.

    _extended_summary_

    Args:
        x: Input data table.
        y: Input decision.
        attrs: A subset of conditional attributes the check should be performed on.
            It should be given in a form of a sequence of integer-location based
            indexing of the selected conditional attributes from ``x``.
        consistent_table_check: Whether decision table consistency check should be
            performed prior to other checks. Defaults to True.

    Returns:
        Indication whether the specified attributes form a reduct.
    """
    if len(set(attrs)) < len(attrs):
        raise ValueError("duplicated attrs in the given sequence")

    if consistent_table_check:
        if not check_if_functional_dependency(x, y):
            return False

    table = np.hstack((x, np.expand_dims(y, axis=1)))
    base_nunique_diff = get_nunique_objs(table) - get_nunique_objs(table[:, :-1])

    xy = np.hstack((x[:, attrs], np.expand_dims(y, axis=1)))
    if base_nunique_diff != get_nunique_objs(xy) - get_nunique_objs(xy[:, :-1]):
        return False

    for i in range(xy.shape[1] - 1):
        xy_no_col = np.delete(xy, i, axis=1)
        nunique_diff = get_nunique_objs(xy_no_col) - get_nunique_objs(xy_no_col[:, :-1])
        if nunique_diff == base_nunique_diff:
            return False

    return True


def check_if_bireduct(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    objs: list[int],
    attrs: list[int],
) -> bool:
    """Check if specified objects and attributes form a bireduct.

    _extended_summary_

    Args:
        x: Input data table.
        x_counts: _description_
        y: Input decision.
        y_count: _description_
        objs: _description_
        attrs: _description_

    Returns:
        Indication whether the specified objects and attributes form a reduct.
    """
    if not check_if_functional_dependency(x, y, objs, attrs):
        return False
    group_index = batch_split_into_groups(x, x_counts, attrs)
    all_objs = np.concatenate((objs, np.arange(len(x))))
    chosen_objs = choose_objects(group_index, y, y_count, all_objs)
    return set(chosen_objs) == set(objs)
