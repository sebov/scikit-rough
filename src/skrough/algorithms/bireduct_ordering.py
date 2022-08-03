import numpy as np

from skrough.checks import check_if_functional_dependency
from skrough.structs.objs_attrs_subset import ObjsAttrsSubset

# attrs_weight = float(ratio) * 2 * .shape[0] / .shape[1]


def get_bireduct_ordering_heuristic(
    x: np.ndarray,
    y: np.ndarray,
    permutation: np.ndarray,
):
    n_objs, n_attrs = x.shape
    n_both = n_objs + n_attrs
    if n_both != len(permutation):
        raise ValueError(
            "length of permutation should match the sum of "
            "both the number of objects and the number of attributes"
        )

    objs = []
    attrs = set(range(n_attrs))
    for i in permutation:
        if i < n_objs:
            ii_obj = i
            objs.append(ii_obj)
            if not check_if_functional_dependency(x, y, objs=objs, attrs=list(attrs)):
                objs.pop()
        else:
            i_attr = i - n_objs
            reduced = attrs - {i_attr}
            if check_if_functional_dependency(x, y, objs=objs, attrs=list(reduced)):
                attrs = reduced
    return ObjsAttrsSubset.from_objs_attrs_like(
        objs_like=sorted(objs),
        attrs_like=sorted(attrs),
    )
