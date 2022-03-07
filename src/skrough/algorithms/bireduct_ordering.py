import numpy as np

import skrough as rgh

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
            if not rgh.checks.check_if_functional_dependency(
                x, y, objs=objs, attrs=list(attrs)
            ):
                objs.pop()
        else:
            ii_attr = ii - nobjs
            reduced = attrs - {ii_attr}
            if rgh.checks.check_if_functional_dependency(
                x, y, objs=objs, attrs=list(reduced)
            ):
                attrs = reduced
    return rgh.containers.Bireduct(objs=sorted(objs), attrs=sorted(attrs))
