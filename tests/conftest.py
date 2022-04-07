import pytest

import skrough as rgh

from . import datasets


@pytest.fixture
def golf_dataset():
    return datasets.golf_dataset()


@pytest.fixture
def golf_dataset_target_attr():
    return "Play"


@pytest.fixture
def golf_dataset_prep(golf_dataset, golf_dataset_target_attr):
    x, x_counts, y, y_count = rgh.dataprep.prepare_factorized_data(
        golf_dataset, target_attr=golf_dataset_target_attr
    )
    return x, x_counts, y, y_count
