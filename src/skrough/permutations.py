from typing import Literal, Optional, Union

import numpy as np

import skrough.typing as rght
from skrough.weights import prepare_weights


def get_permutation(
    start: int,
    stop: int,
    proba: Optional[np.ndarray] = None,
    seed: rght.Seed = None,
) -> np.ndarray:
    """Get permutation.

    Get permutation of values between ``start`` (inclusively) and ``stop`` (exclusively)
    using the given probabilistic distribution ``proba`` for the values being permuted.
    The value of ``proba`` should be understood as follows:
    - when ``proba`` is given as ``np.ndarray`` - it is treated as a disrete
      probabilistic distribution over values from ``start`` to ``stop`` (exclusively)
      and therefore it should sum to 1 and has the length equal to ``stop - start``.
      The higher the probability value on the given position ``i``, the more likely
      for element ``range(start, stop)[i]`` to be earlier in the output permutation.
    - when ``proba is None`` - the uniform distribution is used

    Args:
        start: Start (inclusively) of the interval being permuted.
        stop: Stop (exclusively) of interval being permuted.
        proba: Probabilistic distribution for interval used to create permutation.
            Defaults to None.
        seed: A seed to initialize random Generator. Defaults to None.

    Returns:
        Output permutation.
    """
    if start >= stop:
        result = np.arange(0)
    else:
        rng = np.random.default_rng(seed)
        result = rng.choice(
            np.arange(start, stop),
            size=stop - start,
            replace=False,
            p=proba,
        )
    return result


def get_objs_attrs_permutation(
    nobjs: int,
    nattrs: int,
    objs_weights: Optional[Union[int, float, np.ndarray]] = None,
    attrs_weights: Optional[Union[int, float, np.ndarray]] = None,
    mode: Literal["mixed", "objs_before", "attrs_before"] = "mixed",
    seed: rght.Seed = None,
) -> np.ndarray:
    if mode not in {"mixed", "objs_before", "attrs_before"}:
        raise ValueError("invalid mode argument")

    rng = np.random.default_rng(seed)

    if mode in {"objs_before", "attrs_before"}:
        objs_proba = prepare_weights(
            objs_weights,
            nobjs,
            expand_none=False,
        )
        objs = get_permutation(0, nobjs, objs_proba, seed=rng)
        attrs_proba = prepare_weights(
            attrs_weights,
            nattrs,
            expand_none=False,
        )
        attrs = get_permutation(nobjs, nobjs + nattrs, attrs_proba, seed=rng)
        if mode == "objs_before":
            tmp = (objs, attrs)
        else:
            tmp = (attrs, objs)
        result: np.ndarray = np.concatenate(tmp)
    else:
        weights = np.concatenate(
            (
                prepare_weights(objs_weights, nobjs, normalize=False),
                prepare_weights(attrs_weights, nattrs, normalize=False),
            )
        )
        proba = prepare_weights(weights, nobjs + nattrs)
        result = get_permutation(0, nobjs + nattrs, proba, seed=rng)

    return result


def get_attrs_permutation(
    nattrs: int,
    attrs_weights: Optional[Union[int, float, np.ndarray]] = None,
    seed: rght.Seed = None,
) -> np.ndarray:
    return get_objs_attrs_permutation(
        nobjs=0,
        nattrs=nattrs,
        attrs_weights=attrs_weights,
        mode="attrs_before",
        seed=seed,
    )
