import pathlib

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from skrough.dataprep import (
    add_shuffled_attrs,
    prepare_factorized_array,
    prepare_factorized_vector,
)

MICROARRAY_DATA_DIR = pathlib.Path("../../../../../workspace/data/microarray")
TOOLBOX_DATA_DIR = pathlib.Path(
    "../../../../../workspace/data/toolbox_data/public_data"
)


def _get_data_shuffled(filename, data_dir, sep):
    df = pd.read_csv(
        data_dir / filename,
        index_col=0,
        sep=sep,
    )
    target_attr = df.columns[-1]
    df = add_shuffled_attrs(df, target_attr)
    df_dec = df.pop(target_attr)
    return df, df_dec


def get_microarray_data_shuffled(filename):
    return _get_data_shuffled(filename, MICROARRAY_DATA_DIR, sep=",")


def get_toolbox_data_shuffled(filename):
    return _get_data_shuffled(filename, TOOLBOX_DATA_DIR, sep=",")


def get_discretized_prepared(df, df_dec):
    est = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile")
    df = est.fit_transform(df)
    x, x_counts = prepare_factorized_array(df)
    y, y_count = prepare_factorized_vector(df_dec)
    return x, x_counts, y, y_count
