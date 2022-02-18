from typing import Iterable

import numpy as np
import pandas as pd

from skrough.containers import Reduct


def check_functional_dependency(df, dec, cols, rows):
    """
    Check functional dependency in the dataframe projected to the given
    columns and objects.
    """
    if len(rows) == 0:
        return True
    duplicate_count = (
        (len(rows) - 1) if not cols else df.loc[rows, cols].duplicated().sum()
    )
    duplicate_with_dec_count = df.loc[rows, cols + [dec]].duplicated().sum()
    return duplicate_count == duplicate_with_dec_count


def test_functional_dependency(x, y, objects=None, attributes=None):
    objects = objects if objects is not None else slice(None)
    attributes = attributes if attributes is not None else slice(None)
    if isinstance(objects, Iterable) and isinstance(attributes, Iterable):
        x_index = np.ix_(objects, attributes)
    else:
        x_index = np.index_exp[objects, attributes]
    dfx = pd.DataFrame(x[x_index])
    dfy = pd.DataFrame(y[objects])
    df = pd.concat([dfx, dfy], axis=1)
    if df.shape[0] == 0:
        duplicated = 0
    elif df.shape[1] == 0:
        duplicated = df.shape[0]
    else:
        duplicated = df.iloc[:, :-1].duplicated().sum()
    duplicated_with_dec = df.duplicated().sum()
    return duplicated == duplicated_with_dec


def test_if_reduct(x, y, red):
    # TODO: what if red does not hold functional dependency?
    for i in red.attributes:
        attributes = np.setdiff1d(red.attributes, [i])
        if test_functional_dependency(x, y, attributes=attributes):
            return False
    return True


def test_if_bireduct(x, y, bir):
    xx = x[np.ix_(bir.objects, bir.attributes)]
    yy = y[bir.objects]
    if not test_if_reduct(xx, yy, Reduct(bir.attributes)):
        return False
    else:
        return True


# print(test_if_bireduct(df, df_dec, bir))
