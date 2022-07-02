import numpy as np
import pytest

from skrough.dataprep import prepare_factorized_data
from skrough.predict import predict
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
    predictions = predict(
        model=model,
        reference_data=x,
        reference_data_dec=y,
        predict_data=x,
    )
    np.array_equal(predictions, y)


@pytest.mark.parametrize(
    "x, y, x_test, y_test",
    [
        (
            np.arange(5, 10),
            np.arange(105, 110),
            np.arange(5, 15),
            np.concatenate((np.arange(105, 110), np.repeat(np.nan, 5))),
        ),
    ],
)
def test_predict(x, y, x_test, y_test):
    x = np.expand_dims(np.asarray(x), axis=1)
    y = np.asarray(y)
    x_test = np.expand_dims(np.asarray(x_test), axis=1)
    y_test = np.asarray(y_test)
    model = ObjsAttrsSubset(
        objs=list(range(x.shape[0])),
        attrs=list(range(x.shape[1])),
    )
    predictions = predict(
        model=model,
        reference_data=x,
        reference_data_dec=y,
        predict_data=x_test,
    )
    assert predictions.shape == y_test.shape
    assert np.allclose(predictions, y_test, equal_nan=True)
