import numba
import numba.typed
import numpy as np


@numba.njit
def aggregate_predictions(
    n_objs: int, n_classes: int, predictions_collection: numba.typed.List[np.ndarray]
):
    distribution = np.zeros(
        shape=(n_objs, n_classes),
        dtype=np.float64,
    )

    counts = np.zeros(
        shape=n_objs,
        dtype=np.float64,
    )

    for predictions in predictions_collection:
        for i in range(len(predictions)):  # pylint: disable=consider-using-enumerate
            if not np.isnan(predictions[i]):
                counts[i] += 1
                distribution[i, int(predictions[i])] += 1

    for i in range(n_objs):
        if counts[i] == 0:
            distribution[i, :] = np.nan
        else:
            distribution[i, :] /= counts[i]

    return distribution, counts
