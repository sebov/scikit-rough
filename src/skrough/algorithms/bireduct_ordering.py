import numpy as np

from skrough.checks import check_if_functional_dependency
from skrough.structs import Bireduct

# attrs_weight = float(ratio) * 2 * .shape[0] / .shape[1]


def get_bireduct_ordering_algorithm(
    x: np.ndarray,
    y: np.ndarray,
    permutation: np.ndarray,
):
    nobjs, nattrs = x.shape
    nboth = nobjs + nattrs
    if nboth != len(permutation):
        raise ValueError(
            "length of permutation should match the sum of "
            "both the number of objects and the number of attributes"
        )

    objs = []
    attrs = set(range(nattrs))
    for i in range(nboth):
        ii = permutation[i]
        if ii < nobjs:
            ii_obj = ii
            objs.append(ii_obj)
            if not check_if_functional_dependency(x, y, objs=objs, attrs=list(attrs)):
                objs.pop()
        else:
            ii_attr = ii - nobjs
            reduced = attrs - {ii_attr}
            if check_if_functional_dependency(x, y, objs=objs, attrs=list(reduced)):
                attrs = reduced
    return Bireduct(objs=sorted(objs), attrs=sorted(attrs))
