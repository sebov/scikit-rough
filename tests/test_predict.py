from typing import List, Optional, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from skrough.dataprep import prepare_factorized_data
from skrough.predict import PredictStrategy, predict_objs_attrs
from skrough.structs.objs_attrs_subset import ObjsAttrsSubset


def test_predict_golf(golf_dataset, golf_dataset_target_attr):
    x, _, y, _ = prepare_factorized_data(
        golf_dataset,
        golf_dataset_target_attr,
    )
    model = ObjsAttrsSubset(
        objs=list(range(x.shape[0])),
        attrs=list(range(x.shape[1])),
    )
    predictions = predict_objs_attrs(
        model=model,
        reference_data=x,
        reference_data_y=y,
        predict_data=x,
    )
    np.array_equal(predictions, y)


def run_compare_predict(
    x: np.ndarray,
    y: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    strategy: str,
    objs: Optional[List[int]] = None,
    attrs: Optional[List[int]] = None,
):
    x = np.expand_dims(np.asarray(x), axis=1)
    y = np.asarray(y)
    x_test = np.expand_dims(np.asarray(x_test), axis=1)
    y_test = np.asarray(y_test)
    model = ObjsAttrsSubset(
        objs=objs if objs is not None else list(range(x.shape[0])),
        attrs=attrs if attrs is not None else list(range(x.shape[1])),
    )
    predictions = predict_objs_attrs(
        model=model,
        reference_data=x,
        reference_data_y=y,
        predict_data=x_test,
        strategy=cast(PredictStrategy, strategy),
    )
    assert predictions.shape == y_test.shape
    assert np.allclose(predictions, y_test, equal_nan=True)


@pytest.mark.parametrize(
    "x, y, x_test, y_test",
    [
        (
            np.arange(5, 10),
            np.arange(105, 110),
            np.arange(5, 15),
            np.concatenate((np.arange(105, 110), np.repeat(np.nan, 5))),
        ),
        (
            np.arange(5, 10),
            np.arange(105, 110),
            np.arange(0, 15),
            np.concatenate(
                (
                    np.repeat(np.nan, 5),
                    np.arange(105, 110),
                    np.repeat(np.nan, 5),
                )
            ),
        ),
        (
            [0, 0, 1],
            [0, 1, 2],
            [0, 0, 1],
            [0, 0, 2],
        ),
    ],
)
def test_predict_strategy_original_order(x, y, x_test, y_test):
    run_compare_predict(x, y, x_test, y_test, strategy="original_order")


@pytest.mark.parametrize(
    "x, y, x_test, y_test, permutation",
    [
        (
            [0, 0, 1],
            [0, 1, 2],
            [0, 0, 1, 2],
            [0, 0, 2, np.nan],
            [0, 1, 2],
        ),
        (
            [0, 0, 1],
            [0, 1, 2],
            [0, 0, 1, 2],
            [1, 1, 2, np.nan],
            [1, 0, 2],
        ),
        (
            [0, 0, 1, 1],
            [0, 1, 2, 3],
            [2, 0, 1],
            [np.nan, 0, 2],
            [0, 1, 2, 3],
        ),
        (
            [0, 0, 1, 1],
            [0, 1, 2, 3],
            [2, 0, 1],
            [np.nan, 0, 3],
            [0, 1, 3, 2],
        ),
        (
            [0, 0, 1, 1],
            [0, 1, 2, 3],
            [2, 0, 1],
            [np.nan, 1, 2],
            [1, 0, 2, 3],
        ),
        (
            [0, 0, 1, 1],
            [0, 1, 2, 3],
            [2, 0, 1],
            [np.nan, 1, 3],
            [1, 0, 3, 2],
        ),
    ],
)
@patch("skrough.predict.get_objs_permutation")
def test_predict_strategy_randomized(
    mock: MagicMock,
    x,
    y,
    x_test,
    y_test,
    permutation,
):
    mock.return_value = permutation
    run_compare_predict(x, y, x_test, y_test, strategy="randomized")


@pytest.mark.parametrize("strategy", ["a", "", "qqq", -1])
def test_predict_strategy_unrecognized(strategy):
    with pytest.raises(ValueError, match="Unrecognized"):
        dummy = np.array([])
        run_compare_predict(dummy, dummy, dummy, dummy, strategy=strategy)
