import itertools
import random
import pandas as pd
import sklearn.utils
import logging
from collections.abc import Iterable
import numpy as np
from skrough.base import Reduct, Bireduct


# ------------------------------------------------------------------
def check_functional_dependency(df, dec, cols, rows):
    '''
    Check functional dependency in the dataframe projected to the given
    columns and objects.
    '''
    if len(rows) == 0:
        return True
    duplicate_count = ((len(rows) - 1) if not cols else df.loc[rows, cols].duplicated().sum())
    duplicate_with_dec_count = df.loc[rows, cols + [dec]].duplicated().sum()
    return duplicate_count == duplicate_with_dec_count

def generate_permutation(df, dec, attrs_weight, random_state=None):
    '''
    Generate columns-objects random permutation with regard to the given columns_weight.

    Ratio equals to 0 gives all objects before all columns.

    For dataframe of n columns (not including the decision attribute) by m objects
    the numbering is as follows: numbers 0 to m-1 for columns n to n+m-1 for objects.
    '''
    random_state = sklearn.utils.check_random_state(random_state)
    cols_len = len(df.columns[df.columns != dec])
    if attrs_weight > 0:
        weights = list(itertools.chain(
            itertools.repeat(1, len(df)),
            itertools.repeat(attrs_weight, cols_len)
            ))
        result = pd.Series(range(0, len(weights)))
        result = list(result.sample(len(result), weights=weights, random_state=random_state))
    else:
        result = list(itertools.chain(
            random.sample(range(0, len(df)), len(df)),
            random.sample(range(len(df), len(df) + cols_len), cols_len)
            ))
    return result


def get_bireduct(df, dec, permutation):
    '''
    For a given columns-objects permutation compute a bireduct using
    the ordering algorithm.
    '''
    orig_cols = list(df.columns[df.columns != dec])
    orig_rows = list(df.index)
    cols = set(orig_cols)
    rows = set()
    for i, p in enumerate(permutation):
        if i % 50 == 0:
            logging.debug(f'{i}/{len(permutation)}')
        if p < len(df):
            new_rows = rows | set([orig_rows[p]])
            if check_functional_dependency(df, dec, list(cols), list(new_rows)):
                rows = new_rows
        else:
            new_cols = cols - set([orig_cols[p - len(df)]])
            if check_functional_dependency(df, dec, list(new_cols), list(rows)):
                cols = new_cols
    return (tuple(rows), tuple(cols))


def get_random_bireduct(df, dec, ratio, random_state=None):
    '''
    Generate random bireduct with a given 'ratio' influencing the shuffling
    of columns and objects.
    '''
    random_state = sklearn.utils.check_random_state(random_state)
    attrs_weight = float(ratio) * 2 * df.shape[0] / df.shape[1]
    permutation = generate_permutation(df, dec, attrs_weight, random_state=random_state)
    return get_bireduct(df, dec, permutation)


def get_bireducts_for_ratio(df, dec, ratio, n_bireducts, random_state=None):
    random_state = sklearn.utils.check_random_state(random_state)
    result = []
    for i in range(n_bireducts):
        if i % 50 == 0:
            logging.debug(f'bireduct {i}/{n_bireducts}')
        result.append(get_random_bireduct(df, dec, ratio, random_state))
    return result



# ------------------------------
# def test_functional_dependency(x, y, objects=None, attributes=None):
#     objects = objects if objects is not None else slice(None)
#     attributes = attributes if attributes is not None else slice(None)
#     if isinstance(objects, Iterable) and isinstance(attributes, Iterable):
#     if isinstance(objects, Iterable) and isinstance(attributes, Iterable):
#         x_index = np.ix_(objects, attributes)
#     else:
#         x_index = np.index_exp[objects, attributes]
#     dfx = pd.DataFrame(x[x_index])
#     dfy = pd.DataFrame(y[objects])
#     df = pd.concat([dfx, dfy], axis=1)
#     if df.shape[0] == 0:
#         duplicated = 0
#     elif df.shape[1] == 0:
#         duplicated = df.shape[0]
#     else:
#         duplicated = df.iloc[:, :-1].duplicated().sum()
#     duplicated_with_dec = df.duplicated().sum()
#     return duplicated == duplicated_with_dec
#
# def test_if_reduct(x, y, red):
#     # TODO: what if red does not hold functional dependency?
#     for i in red.attributes:
#         attributes = np.setdiff1d(red.attributes, [i])
#         if test_functional_dependency(x, y, attributes=attributes):
#             return False
#     return True
#
# def test_if_bireduct(x, y, bir):
#     xx = x[np.ix_(bir.objects, bir.attributes)]
#     yy = y[bir.objects]
#     if not test_if_reduct(xx, yy, Reduct(bir.attributes)):
#         return False
#     else:
#         return True
