import numpy as np

from skrough.algorithms.hooks.names import (
    VALUES_X,
    VALUES_X_COUNTS,
    VALUES_Y,
    VALUES_Y_COUNT,
)
from skrough.dataprep import prepare_factorized_array, prepare_factorized_vector
from skrough.structs.state import ProcessingState


def prepare_test_data_and_setup_state(x, y, state: ProcessingState):
    x, x_counts = prepare_factorized_array(np.asarray(x))
    y, y_count = prepare_factorized_vector(np.asarray(y))
    state.values = {
        VALUES_X: x,
        VALUES_X_COUNTS: x_counts,
        VALUES_Y: y,
        VALUES_Y_COUNT: y_count,
    }
    return x, x_counts, y, y_count, state
