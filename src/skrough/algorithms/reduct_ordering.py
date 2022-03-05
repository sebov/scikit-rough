import numpy as np

import skrough as rgh


def get_reduct_ordering_algorithm(
    x: np.ndarray,
    y: np.ndarray,
    permutation: np.ndarray,
):
    nattrs = x.shape[1]
    if nattrs != len(permutation):
        raise ValueError("length of permutation should match the number of objects")
    result = set(range(nattrs))
    for i in range(nattrs):
        reduced = result - {permutation[i]}
        if rgh.checks.check_if_functional_dependency(x, y, attrs=list(reduced)):
            result = reduced
    return rgh.containers.Reduct(attrs=sorted(result))
