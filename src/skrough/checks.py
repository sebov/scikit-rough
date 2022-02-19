from typing import Any, Optional, Sequence

import numpy as np


def get_nunique_objs(x):
    if x.shape[0] == 0:
        nunique = len(x)
    elif x.shape[1] == 0:
        nunique = 1
    else:
        nunique = len(np.unique(x, axis=0))
    return nunique


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
    nunique = get_nunique_objs(xx)
    nunique_with_dec = get_nunique_objs(xxyy)
    return nunique == nunique_with_dec


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
