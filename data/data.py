from dataclasses import dataclass
from pathlib import Path
from typing import Union

import pandas as pd

DATA_DIR = Path(__file__).parent / "resources"


@dataclass
class Dataset:
    data: pd.DataFrame
    target_col: Union[int, str]


def get_golf_dataset():
    df = pd.read_csv(DATA_DIR / "golf.csv")
    dec_col = "Play"
    return Dataset(df, dec_col)


def get_zoo_dataset():
    df = pd.read_csv(DATA_DIR / "zoo.data", header=None)
    df.drop(0, axis=1, inplace=True)
    dec_col = 17
    return Dataset(df, dec_col)


def get_lymphography_dataset():
    df = pd.read_csv(DATA_DIR / "lymphography.data", header=None)
    df_class = df[0]
    df.drop(0, axis=1, inplace=True)
    df[19] = df_class
    dec_col = 19
    return Dataset(df, dec_col)


def get_data_methane():
    df = pd.read_csv(DATA_DIR / "methane_data.csv", sep=";")
    df_target = pd.read_csv(DATA_DIR / "methane_labels.csv")
    df = pd.concat([df, df_target], axis=1)
    df = df.astype("category")
    df = df.apply(lambda x: x.cat.codes)
    df = df.loc[:, df.nunique() != 1]
    return Dataset(df, "MM263")
