"""Cross-implementation consistency tests for all GroupIndex variants.

Verifies that every registered GroupIndex implementation produces
identical disorder scores and equivalent distributions for the same
input data.
"""

import numpy as np
import pytest

from skrough.dataprep import prepare_factorized_array, prepare_factorized_vector
from skrough.disorder_measures import conflicts_count, entropy, gini_impurity
from skrough.disorder_score import get_disorder_score_for_data
from skrough.structs.group_index import GROUP_INDEX_BY_NAME, GroupIndex
from tests.helpers import generate_data


ALL_IMPLEMENTATIONS = list(GROUP_INDEX_BY_NAME.values())


DATASETS = [
    np.zeros(shape=(4, 3), dtype=np.int64),
    np.ones(shape=(10, 10), dtype=np.int64),
    generate_data(size=(6, 4)),
    np.eye(5, dtype=np.int64),
    np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=np.int64,
    ),
    generate_data(size=(20, 6)),
    generate_data(size=(50, 8)),
]


@pytest.mark.parametrize("disorder_measure", [conflicts_count, entropy, gini_impurity])
@pytest.mark.parametrize("data", DATASETS)
def test_get_disorder_score_consistency(data, disorder_measure):
    data = np.asarray(data)
    x, x_counts = prepare_factorized_array(data[:, 0:-1])
    y, y_count = prepare_factorized_vector(data[:, -1])

    scores = {}
    for gi_class in ALL_IMPLEMENTATIONS:
        group_index = gi_class.from_data(x, x_counts)
        score = group_index.get_disorder_score(y, y_count, disorder_measure)
        if gi_class is GroupIndex:
            scores["reference"] = score
        scores[gi_class.__name__] = score

    ref = scores.pop("reference")
    for name, score in scores.items():
        assert score == ref, f"{name} score {score} != reference {ref}"


@pytest.mark.parametrize("disorder_measure", [conflicts_count, entropy, gini_impurity])
@pytest.mark.parametrize("data", DATASETS)
def test_get_disorder_score_after_split_consistency(data, disorder_measure):
    data = np.asarray(data)
    x, x_counts = prepare_factorized_array(data[:, 0:-2])
    split_values, split_values_count = prepare_factorized_vector(data[:, -2])
    y, y_count = prepare_factorized_vector(data[:, -1])

    scores = {}
    for gi_class in ALL_IMPLEMENTATIONS:
        group_index = gi_class.from_data(x, x_counts)
        score = group_index.get_disorder_score_after_split(
            split_values=split_values,
            split_values_count=split_values_count,
            values=y,
            values_count=y_count,
            disorder_fun=disorder_measure,
        )
        if gi_class is GroupIndex:
            scores["reference"] = score
        scores[gi_class.__name__] = score

    ref = scores.pop("reference")
    for name, score in scores.items():
        assert score == ref, f"{name} score {score} != reference {ref}"


@pytest.mark.parametrize("data", DATASETS)
def test_get_distribution_consistency(data):
    data = np.asarray(data)
    x, x_counts = prepare_factorized_array(data[:, 0:-1])
    y, y_count = prepare_factorized_vector(data[:, -1])

    ref_rows = None
    ref_n_groups = None
    for gi_class in ALL_IMPLEMENTATIONS:
        group_index = gi_class.from_data(x, x_counts)
        distribution = group_index.get_distribution(y, y_count)

        assert distribution.shape[1] == y_count
        assert distribution.sum() == len(y)

        row_set = frozenset(tuple(row) for row in distribution.tolist())

        if ref_rows is None:
            ref_rows = row_set
            ref_n_groups = group_index.n_groups
        else:
            assert row_set == ref_rows, (
                f"{gi_class.__name__} distribution rows differ from reference"
            )


@pytest.mark.parametrize("n_objs", [1, 2, 5, 10])
def test_create_uniform_consistency(n_objs):
    y = np.zeros(n_objs, dtype=np.int64)
    y_count = 1

    for gi_class in ALL_IMPLEMENTATIONS:
        group_index = gi_class.create_uniform(n_objs)
        if n_objs == 0:
            assert group_index.n_objs == 0
        else:
            assert group_index.n_objs == n_objs
            score = group_index.get_disorder_score(y, y_count, entropy)
            assert score == 0.0


@pytest.mark.parametrize("n_objs", [1, 2, 5, 10])
def test_split_chain_consistency(n_objs):
    rng = np.random.default_rng(42)
    data = rng.integers(0, 3, size=(n_objs, 4)).astype(np.int64)
    x, x_counts = prepare_factorized_array(data[:, 0:-1])
    y, y_count = prepare_factorized_vector(data[:, -1])

    split_values, sv_count = prepare_factorized_vector(data[:, 1])

    for disorder_measure in [conflicts_count, entropy, gini_impurity]:
        scores = {}
        for gi_class in ALL_IMPLEMENTATIONS:
            group_index = gi_class.from_data(x, x_counts, attrs=[0])
            group_index = group_index.split(split_values, sv_count)
            score = group_index.get_disorder_score(y, y_count, disorder_measure)
            if gi_class is GroupIndex:
                scores["reference"] = score
            scores[gi_class.__name__] = score

        ref = scores.pop("reference")
        for name, score in scores.items():
            assert score == ref, f"{name} score {score} != reference {ref}"
