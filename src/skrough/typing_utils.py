import numpy as np

import skrough.typing as rght


def unify_objs(objs: rght.ObjsLike) -> rght.Objs:
    return np.asarray(objs, dtype=np.int64)


def unify_attrs(attrs: rght.AttrsLike) -> rght.Attrs:
    return np.asarray(attrs, dtype=np.int64)
