import numpy as np

from skrough.checks import check_if_functional_dependency
from skrough.structs.attrs_subset import AttrsSubset


def get_reduct_ordering_heuristic(
    x: np.ndarray,
    y: np.ndarray,
    permutation: np.ndarray,
) -> AttrsSubset:
    n_attrs = x.shape[1]
    if n_attrs != len(permutation):
        raise ValueError("length of permutation should match the number of objects")
    result = set(range(n_attrs))
    for i in range(n_attrs):
        reduced = result - {permutation[i]}
        if check_if_functional_dependency(x, y, attrs=list(reduced)):
            result = reduced
    return AttrsSubset.from_attrs_like(attrs_subset_like=sorted(result))
