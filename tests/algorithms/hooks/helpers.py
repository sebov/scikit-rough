import numpy as np

from skrough.dataprep import prepare_factorized_array, prepare_factorized_vector
from skrough.structs.state import ProcessingState


def prepare_test_data_and_setup_state(x, y, state: ProcessingState):
    x, x_counts = prepare_factorized_array(np.asarray(x))
    y, y_count = prepare_factorized_vector(np.asarray(y))
    state.set_values_x(x)
    state.set_values_x_counts(x_counts)
    state.set_values_y(y)
    state.set_values_y_count(y_count)
    return x, x_counts, y, y_count, state
