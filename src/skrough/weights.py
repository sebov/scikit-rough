from typing import Optional, Union

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


def prepare_weights(
    weights: Optional[Union[int, float, np.ndarray]],
    n: int,
    normalize: bool = True,
    expand_none: bool = True,
) -> Optional[np.ndarray]:
    if weights is None:
        if expand_none:
            weights = 1.0
        else:
            return None
    if isinstance(weights, (int, float)):
        weights = np.repeat(weights, n)
    if normalize:
        weights = normalize_weights(weights)
    return weights
