import pathlib
import warnings

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from skrough.dataprep import (
    add_shuffled_attrs,
    prepare_factorized_array,
    prepare_factorized_vector,
)

MICROARRAY_DATA_DIR = pathlib.Path("../../../../../workspace/data/microarray")
SYNTHETIC_DATA_DIR = pathlib.Path("../../../../../workspace/data/synthetic_phd")
TOOLBOX_DATA_DIR = pathlib.Path(
    "../../../../../workspace/data/toolbox_data/public_data"
)


def get_microarray_data_shuffled(filename, data_dir=MICROARRAY_DATA_DIR, sep=","):
    df = pd.read_csv(
        data_dir / filename,
        index_col=0,
        sep=sep,
    )
    target_attr = df.columns[-1]
    df = add_shuffled_attrs(df, target_attr)
    df_dec = df.pop(target_attr)
    return df, df_dec


def get_synthetic_data_shuffled(filename, data_dir=SYNTHETIC_DATA_DIR, sep=","):
    df = pd.read_csv(
        data_dir / filename,
        sep=sep,
    )
    target_attr = df.columns[-1]
    df = add_shuffled_attrs(df, target_attr)
    df_dec = df.pop(target_attr).astype("int")
    return df, df_dec


def get_toolbox_data_shuffled(filename, data_dir=TOOLBOX_DATA_DIR, sep=","):
    df = pd.read_csv(
        data_dir / filename,
        index_col=0,
        sep=sep,
    )
    df.drop("process_id", axis=1, inplace=True)
    target_attr = df.columns[-1]
    df = add_shuffled_attrs(df, target_attr)
    df_dec = df.pop(target_attr).astype("category").cat.codes
    df_dec = 1 - df_dec

    est = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile")
    cols_to_discretize = df.nunique() > 3
    kbin = est.fit_transform(df.loc[:, cols_to_discretize])
    df[df.columns[cols_to_discretize]] = kbin
    df = df.astype("category")
    df = df.apply(lambda x: x.cat.codes)

    return df, df_dec


def get_discretized_prepared(df, df_dec):
    est = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile")
    df = est.fit_transform(df)
    x, x_counts = prepare_factorized_array(df)
    y, y_count = prepare_factorized_vector(df_dec)
    return x, x_counts, y, y_count
