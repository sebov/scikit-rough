from typing import Any, Optional, Sequence

import numpy as np


def get_nunique_objs(x: np.ndarray) -> int:
    """Compute the number of unique rows.

    Compute the number of unique rows. Degenerated tables are handled accordingly,
    i.e., a table with no columns has 1 unique rows if only it has at least one row,
    otherwise it is 0.

    Parameters
    ----------
    x
        Input data table.

    Returns
    -------
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

    Parameters
    ----------
    x
        Input data table.
    y
        Input decision.
    objs: optional, default=None
        A subset of object that the check should be performed on. It should
        be given in a form of a sequence of integer-location based indexing of the
        selected objects/rows/instances from ``x``. ``None`` value means to use
        all available objects.
    attrs: optional, default=None
        A subset of conditional attributes the check should be performed on. It should
        be given in a form of a sequence of integer-location based indexing of the
        selected conditional attributes from ``x``. ``None`` value means to use
        all available conditional attributes.

    Returns
    -------
        Indication whether functional dependency holds for the given input.
    """
    objects = objs if objs is not None else slice(None)
    attributes = attrs if attrs is not None else slice(None)
    xx_index_expr: Any
    if not (isinstance(objects, Sequence) and isinstance(attributes, Sequence)):
        xx_index_expr = np.index_exp[objects, attributes]
    else:
        # we want to take all ``objects`` x ``attributes``
        xx_index_expr = np.ix_(objects, attributes)
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
    just as a simple wrapper around ``check_if_functional_dependency`` function with all
    using all available objects and attributes.

    Parameters
    ----------
    x
        Input data table.
    y
        Input decision

    Returns
    -------
        Indication whether the decision table is consistent.
    """
    return check_if_functional_dependency(x, y)


def check_if_reduct(
    x: np.ndarray,
    y: np.ndarray,
    attrs: list[int],
    consistent_table_check: bool = True,
) -> bool:
    """Check if given attrs form a reduct.

    _extended_summary_

    Parameters
    ----------
    x
        Input data table.
    y
        Input decision.
    attrs
        A subset of conditional attributes the check should be performed on. It should
        be given in a form of a sequence of integer-location based indexing of the
        selected conditional attributes from ``x``.
    consistent_table_check: optional, default=True
        Whether decision table consistency check should be performed prior to other
        checks.

    Returns
    -------
        Indication whether the given subset of attributes are a reduct.

    Raises
    ------
    Exception
        _description_
    """
    if len(set(attrs)) < len(attrs):
        raise Exception("duplicated attrs in the given sequence")

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


# def test_if_bireduct(x, y, bir):
#     xx = x[np.ix_(bir.objects, bir.attributes)]
#     yy = y[bir.objects]
#     if not test_if_reduct(xx, yy, Reduct(bir.attributes)):
#         return False
#     else:
#         return True
