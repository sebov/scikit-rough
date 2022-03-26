import numpy as np


def get_uniques_index(values: np.ndarray) -> np.ndarray:
    _, idx = np.unique(values, return_index=True)
    return idx
