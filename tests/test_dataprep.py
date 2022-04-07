import numpy as np
import pandas as pd

import skrough as rgh


def test_prepare_factorized_data(
    golf_dataset: pd.DataFrame,
    golf_dataset_target_attr: str,
):
    x, x_counts, y, y_count = rgh.dataprep.prepare_factorized_data(
        golf_dataset, target_attr=golf_dataset_target_attr
    )
    assert np.array_equal(pd.DataFrame(x).nunique().to_numpy(), x_counts)
    assert pd.Series(y).nunique() == y_count


def test_add_shadow_attrs(
    golf_dataset: pd.DataFrame,
    golf_dataset_target_attr: str,
):
    shadow_attrs_prefix = "shadow_"
    shadow_golf_dataset = rgh.dataprep.add_shadow_attrs(
        df=golf_dataset,
        target_attr=golf_dataset_target_attr,
        shadow_attrs_prefix=shadow_attrs_prefix,
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
