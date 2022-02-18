import itertools
import random
from typing import Union

import pandas as pd
import sklearn.utils

from skrough.typing import RandomStateType


def generate_permutation(
    df: pd.DataFrame,
    target_column: Union[str, int],
    attrs_weight: float,
    random_state: RandomStateType = None,
):
    """
    Generate columns-objects random permutation with regard to the given columns_weight.

    Ratio equals to 0 gives all objects before all columns.

    For dataframe of n columns (not including the decision attribute) by m objects
    the numbering is as follows: numbers 0 to m-1 for columns n to n+m-1 for objects.
    """
    random_state = sklearn.utils.check_random_state(random_state)
    cols_len = len(df.columns[df.columns != target_column])
    if attrs_weight > 0:
        weights = list(
            itertools.chain(
                itertools.repeat(1, len(df)), itertools.repeat(attrs_weight, cols_len)
            )
        )
        result = pd.Series(range(0, len(weights)))
        result = list(
            result.sample(len(result), weights=weights, random_state=random_state)
        )
    else:
        result = list(
            itertools.chain(
                random.sample(range(0, len(df)), len(df)),
                random.sample(range(len(df), len(df) + cols_len), cols_len),
            )
        )
    return result
