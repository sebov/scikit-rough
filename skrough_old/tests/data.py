import pandas as pd
import os


DATA_DIR = 'resources'


def get_golf_dataset():
    df = pd.read_csv(os.path.join(DATA_DIR, 'golf.csv'))
    dec_col = 'Play'
    return df, dec_col


def get_zoo_dataset():
    df = pd.read_csv('zoo.data', header=None)
    df.index = range(1, len(df) + 1)
    df.drop(0, axis=1, inplace=True)
    dec_col = 17
    return df, dec_col


def get_lymphography_dataset():
    df = pd.read_csv(os.path.join(DATA_DIR, 'lymphography.data'), header=None)
    df_class = df[0]
    df.drop(0, inplace=True)
    df[19] = df_class
    dec_col = 19
    return df, dec_col



