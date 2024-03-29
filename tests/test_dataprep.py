import numpy as np
import pandas as pd
import pytest

import skrough as rgh
from skrough.dataprep import prepare_factorized_array, prepare_factorized_vector
from tests.helpers import generate_data


def test_prepare_factorized_data(
    golf_dataset: pd.DataFrame,
    golf_dataset_target_attr: str,
):
    x, x_counts, y, y_count = rgh.dataprep.prepare_factorized_data(
        golf_dataset, target_attr=golf_dataset_target_attr
    )
    assert np.array_equal(pd.DataFrame(x).nunique().to_numpy(), x_counts)
    assert pd.Series(y).nunique() == y_count


@pytest.mark.parametrize(
    "values, expected, expected_count",
    [
        ([], [], 0),
        ([0], [0], 1),
        ([2], [0], 1),
        ([-3], [0], 1),
        ([0, 1, 2], [0, 1, 2], 3),
        ([1, 2, 10], [0, 1, 2], 3),
        ([-1, -2, -3], [0, 1, 2], 3),
        ([1, 4, -10], [0, 1, 2], 3),
        ([1, 1, -10, 0], [0, 0, 1, 2], 3),
        ([1, 1, 2, 3, 2, 10, 1], [0, 0, 1, 2, 1, 3, 0], 4),
        ([1, 1, 1, 1], [0, 0, 0, 0], 1),
        ([-1, 1, -1, 1], [0, 1, 0, 1], 2),
    ],
)
def test_prepare_factorized_values(values, expected, expected_count):
    values = np.asarray(values)
    expected = np.asarray(expected)
    result, result_count = prepare_factorized_vector(values)
    assert result_count == expected_count
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "array, expected, expected_counts",
    [
        (generate_data(size=(0, 0)), generate_data(size=(0, 0)), []),
        (generate_data(size=(1, 0)), generate_data(size=(1, 0)), []),
        (generate_data(size=(2, 0)), generate_data(size=(2, 0)), []),
        (generate_data(size=(0, 1)), generate_data(size=(0, 1)), [0]),
        (generate_data(size=(0, 2)), generate_data(size=(0, 2)), [0, 0]),
        ([[0]], [[0]], [1]),
        ([[2]], [[0]], [1]),
        ([[-2]], [[0]], [1]),
        ([[0, 1], [1, 0]], [[0, 0], [1, 1]], [2, 2]),
        (
            [
                [0, 1, 2],
                [1, 1, 1],
                [2, 1, 0],
            ],
            [
                [0, 0, 0],
                [1, 0, 1],
                [2, 0, 2],
            ],
            [3, 1, 3],
        ),
    ],
)
def test_prepare_factorized_array(array, expected, expected_counts):
    array = np.asarray(array)
    expected = np.asarray(expected)
    result, result_counts = prepare_factorized_array(array)
    assert np.array_equal(result_counts, expected_counts)
    assert np.array_equal(result, expected)


def test_add_shadow_attrs(
    golf_dataset: pd.DataFrame,
    golf_dataset_target_attr: str,
):
    shadow_attrs_prefix = "shadow_"
    shadow_golf_dataset = rgh.dataprep.add_shuffled_attrs(
        df=golf_dataset,
        target_attr=golf_dataset_target_attr,
        shuffled_attrs_prefix=shadow_attrs_prefix,
    )
    conditional_attrs = [
        attr for attr in golf_dataset.columns if attr != golf_dataset_target_attr
    ]
    conditional_attrs_count = len(conditional_attrs)
    # shadow_dataset should have columns count equal to
    # 2 * conditional_attrs_count (orig + shadow for each column) + target attr
    assert shadow_golf_dataset.shape[1] == 2 * conditional_attrs_count + 1

    for attr in conditional_attrs:
        assert attr in shadow_golf_dataset
        shadow_attr = f"{shadow_attrs_prefix}{attr}"
        assert shadow_attr in shadow_golf_dataset

        orig_values = golf_dataset[attr]
        shadow_orig_values = shadow_golf_dataset[attr]
        shadow_shadow_values = shadow_golf_dataset[shadow_attr]

        assert orig_values.equals(shadow_orig_values)
        assert np.array_equal(
            orig_values.sort_values(),
            shadow_shadow_values.sort_values(),
        )
