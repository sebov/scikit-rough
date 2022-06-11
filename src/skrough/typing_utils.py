import numpy as np

import skrough.typing as rght


def unify_objs(objs: rght.Objs) -> rght.UnifiedObjs:
    return np.asarray(objs)


def unify_attrs(attrs: rght.Attrs) -> rght.UnifiedAttrs:
    return np.asarray(attrs)
