import numpy as np


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    def normalize(values: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(values, ord=1)
        if norm > 0:
            values = values / norm
        return values

    weights = np.asarray(weights)
    weights = normalize(weights)
    if any(weights == 0):
        weights += np.finfo(dtype=np.float64).eps
        weights = normalize(weights)

    return weights
