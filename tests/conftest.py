import pytest

import skrough as rgh

from . import datasets


@pytest.fixture
def golf_dataset():
    return datasets.golf_dataset()


@pytest.fixture
def golf_dataset_prep(golf_dataset):
    x, x_counts, y, y_count = rgh.dataprep.prepare_df(
        golf_dataset, target_column="Play"
    )
    return x, x_counts, y, y_count
