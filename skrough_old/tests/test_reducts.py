'''Test reducts'''

import os
import pandas as pd
from skrough.reducts.greedy_heuristic_reduct import GreedyHeuristicReduct


DATA_DIR = 'resources'


def test_methane():
    data_filename = os.path.join(DATA_DIR, 'methane_data.csv')
    labels_filename = os.path.join(DATA_DIR, 'methane_labels.csv')

    df = pd.read_csv(data_filename, sep=';')
    df = df.astype('category')
    df = df.apply(lambda x: x.cat.codes)
    labels = pd.read_csv(labels_filename, sep=';')
    labels = labels.astype('category')
    labels = labels.apply(lambda x: x.cat.codes)
    labels = labels["MM263"]

    ghr = GreedyHeuristicReduct()
    ghr = ghr.fit(df, labels, check_data_consistency=False)
    red = ghr.get_reduct()



