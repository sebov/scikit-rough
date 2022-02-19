from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd


def check_functional_dependency_slower(x, y, objects=None, attributes=None):
    objects = objects if objects is not None else slice(None)
    attributes = attributes if attributes is not None else slice(None)
    if isinstance(objects, Sequence) and isinstance(attributes, Sequence):
        x_index = np.ix_(objects, attributes)
    else:
        x_index = np.index_exp[objects, attributes]
    dfx = pd.DataFrame(x[x_index])
    dfy = pd.DataFrame(y[objects])
    df = pd.concat([dfx, dfy], axis=1)
    if df.shape[0] == 0:
        duplicated = 0
    elif df.shape[1] == 1:
        duplicated = df.shape[0] - 1
    else:
        duplicated = df.iloc[:, :-1].duplicated().sum()
    duplicated_with_dec = df.duplicated().sum()
    return duplicated == duplicated_with_dec


def check_functional_dependency(
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
        input data table
    y
        input decision
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
    if xx.shape[0] == 0:
        duplicated = 0
    elif xx.shape[1] == 0:
        duplicated = xx.shape[0] - 1
    else:
        duplicated = len(xx) - len(np.unique(xx, axis=0))
    duplicated_with_dec = len(xxyy) - len(np.unique(xxyy, axis=0))
    return duplicated == duplicated_with_dec


# def check_functional_dependency(df, dec, cols, rows):
#     """
#     Check functional dependency in the dataframe projected to the given
#     columns and objects.
#     """
#     if len(rows) == 0:
#         return True
#     duplicate_count = (
#         (len(rows) - 1) if not cols else df.loc[rows, cols].duplicated().sum()
#     )
#     duplicate_with_dec_count = df.loc[rows, cols + [dec]].duplicated().sum()
#     return duplicate_count == duplicate_with_dec_count


# def test_if_reduct(x, y, red):
#     # TODO: what if red does not hold functional dependency?
#     for i in red.attributes:
#         attributes = np.setdiff1d(red.attributes, [i])
#         if test_functional_dependency(x, y, attributes=attributes):
#             return False
#     return True


# def test_if_bireduct(x, y, bir):
#     xx = x[np.ix_(bir.objects, bir.attributes)]
#     yy = y[bir.objects]
#     if not test_if_reduct(xx, yy, Reduct(bir.attributes)):
#         return False
#     else:
#         return True
