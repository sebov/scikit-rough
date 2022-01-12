import numpy as np
import pandas as pd

import skrough as rgh


def test_prepare_df(golf_dataset):
    x, x_counts, y, y_count = rgh.dataprep.prepare_df(
        golf_dataset, target_column="Play"
    )
    assert np.array_equal(pd.DataFrame(x).nunique().to_numpy(), x_counts)
    assert pd.Series(y).nunique() == y_count
