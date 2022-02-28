from typing import Literal, Optional, Union

import numpy as np

import skrough as rgh
import skrough.typing as rght


def draw_values(
    low: int,
    high: int,
    proba: Optional[np.ndarray] = None,
    seed: rght.RandomState = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(
        np.arange(low, high),
        size=high - low,
        replace=False,
        p=proba,
    )


def get_objs_attrs_permutation(
    nobjs: int,
    nattrs: int,
    mode: Literal["mixed", "objs_before", "attrs_before"] = "mixed",
    objs_weights: Optional[Union[int, float, np.ndarray]] = None,
    attrs_weights: Optional[Union[int, float, np.ndarray]] = None,
    seed: rght.RandomState = None,
):
    if mode not in {"mixed", "objs_before", "attrs_before"}:
        raise ValueError("invalid mode argument")

    rng = np.random.default_rng(seed)

    if mode in {"objs_before", "attrs_before"}:
        objs_proba = rgh.weights.prepare_weights(
            objs_weights,
            nobjs,
            expand_none=False,
        )
        objs = draw_values(0, nobjs, objs_proba, seed=rng)
        attrs_proba = rgh.weights.prepare_weights(
            attrs_weights,
            nattrs,
            expand_none=False,
        )
        attrs = draw_values(nobjs, nobjs + nattrs, attrs_proba, seed=rng)
        if mode == "objs_before":
            tmp = (objs, attrs)
        else:
            tmp = (attrs, objs)
        result = np.concatenate(tmp)
    else:
        weights = np.concatenate(
            (
                rgh.weights.prepare_weights(objs_weights, nobjs, normalize=False),
                rgh.weights.prepare_weights(attrs_weights, nattrs, normalize=False),
            )
        )
        proba = rgh.weights.prepare_weights(weights, nobjs + nattrs)
        result = draw_values(0, nobjs + nattrs, proba, seed=rng)

    return result
