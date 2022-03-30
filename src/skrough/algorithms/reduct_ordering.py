import numpy as np

from skrough.checks import check_if_functional_dependency
from skrough.containers import Reduct


def get_reduct_ordering_algorithm(
    x: np.ndarray,
    y: np.ndarray,
    permutation: np.ndarray,
) -> Reduct:
    nattrs = x.shape[1]
    if nattrs != len(permutation):
        raise ValueError("length of permutation should match the number of objects")
    result = set(range(nattrs))
    for i in range(nattrs):
        reduced = result - {permutation[i]}
        if check_if_functional_dependency(x, y, attrs=list(reduced)):
            result = reduced
    return Reduct(attrs=sorted(result))
