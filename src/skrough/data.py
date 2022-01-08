import numpy as np
import pandas as pd
import sklearn.utils


def _prepare_values(values):
    """
    Prepare/factorize values
    """
    factorized_values, uniques = pd.factorize(values, na_sentinel=None)  # type: ignore
    uniques = len(uniques)
    return factorized_values, uniques


def prepare_df(df, target_column):
    """
    Prepare/factorize data table
    """
    y = df[target_column]
    x = df.drop(columns=target_column)
    data = x.apply(_prepare_values, 0)
    x = np.vstack(data.values[0]).T  # type: ignore
    x_count_distinct = data.values[1].astype(int)
    y, y_count_distinct = _prepare_values(y)
    x, y = sklearn.utils.check_X_y(x, y, multi_output=False)
    return x, x_count_distinct, y, y_count_distinct
