import numpy as np

from skrough.dataprep import prepare_factorized_data
from skrough.predict import predict
from skrough.structs.objs_attrs_subset import ObjsAttrsSubset


def test_predict(golf_dataset, golf_dataset_target_attr):
    x, x_counts, y, y_count = prepare_factorized_data(
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
        input_data=x,
    )
    np.array_equal(predictions, y)
