import sys
import numpy as np
import pathlib
import pandas as pd
import skrough.reducts.greedy_heuristic_reduct
import skrough.bireducts.sampling_heuristic_bireduct


def get_data_methane():
    DATA_DIR = pathlib.Path(pathlib.Path.home(), 'workspace', 'data')
    train_data = pathlib.Path(DATA_DIR, 'discretizedMethaneData1_freqDisc_training.csv')
    train_labels = pathlib.Path(DATA_DIR, 'trainingLabels.csv')
    df_methane = pd.read_csv(train_data, sep=';')
    df_methane = df_methane.astype('category')
    df_methane = df_methane.apply(lambda x: x.cat.codes)
    counts = df_methane.apply(pd.Series.nunique)
    df_methane = df_methane.loc[:, counts != 1]
    df_methane_labels = pd.read_csv(train_labels, sep=';')
    df_methane_labels = df_methane_labels.astype('category')
    df_methane_labels = df_methane_labels.apply(lambda x: x.cat.codes)
    dec_methane = df_methane_labels["MM263"]
    return df_methane, dec_methane

df_methane, dec_methane = get_data_methane()


