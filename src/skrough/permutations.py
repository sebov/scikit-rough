"""Permutation related utils."""

from typing import Literal, Mapping, Optional, Union, get_args

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

    - when ``proba`` is given in a form of :class:`numpy.ndarray` - it is treated as a
      discrete probabilistic distribution over values from ``start`` to ``stop``
      (exclusively) and therefore it should sum to ``1.0`` and has the length equal to
      :code:`stop - start`. The higher the probability value on the given position
      ``i``, the more likely for element :code:`range(start, stop)[i]` to appear earlier
      in the output permutation.
    - when :code:`proba is None` - the uniform distribution is used

    Args:
        start: Start (inclusively) of the interval being permuted.
        stop: Stop (exclusively) of interval being permuted.
        proba: Probabilistic distribution for interval used to create permutation.
            Defaults to :obj:`None`.
        seed: A seed to initialize random generator. Defaults to :obj:`None`.

    Returns:
        Output permutation.
    """
    if start >= stop:
        result = np.arange(0, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        result = rng.choice(
            np.arange(start, stop),
            size=stop - start,
            replace=False,
            p=proba,
        )
    return result


def _get_objs_attrs_permutation_strategy_one_before(
    objs_before: bool,
    n_objs: int,
    n_attrs: int,
    objs_weights: Optional[Union[int, float, np.ndarray]] = None,
    attrs_weights: Optional[Union[int, float, np.ndarray]] = None,
    rng: rght.Seed = None,
) -> np.ndarray:
    objs_proba = prepare_weights(
        objs_weights,
        n_objs,
        expand_none=False,
    )
    objs = get_permutation(0, n_objs, objs_proba, seed=rng)
    attrs_proba = prepare_weights(
        attrs_weights,
        n_attrs,
        expand_none=False,
    )
    attrs = get_permutation(n_objs, n_objs + n_attrs, attrs_proba, seed=rng)
    if objs_before:
        tmp = (objs, attrs)
    else:
        tmp = (attrs, objs)
    result = np.concatenate(tmp)
    return result


def get_objs_attrs_permutation_strategy_objs_before(
    n_objs: int,
    n_attrs: int,
    objs_weights: Optional[Union[int, float, np.ndarray]] = None,
    attrs_weights: Optional[Union[int, float, np.ndarray]] = None,
    rng: rght.Seed = None,
) -> np.ndarray:
    return _get_objs_attrs_permutation_strategy_one_before(
        objs_before=True,
        n_objs=n_objs,
        n_attrs=n_attrs,
        objs_weights=objs_weights,
        attrs_weights=attrs_weights,
        rng=rng,
    )


def get_objs_attrs_permutation_strategy_attrs_before(
    n_objs: int,
    n_attrs: int,
    objs_weights: Optional[Union[int, float, np.ndarray]] = None,
    attrs_weights: Optional[Union[int, float, np.ndarray]] = None,
    rng: rght.Seed = None,
) -> np.ndarray:
    return _get_objs_attrs_permutation_strategy_one_before(
        objs_before=False,
        n_objs=n_objs,
        n_attrs=n_attrs,
        objs_weights=objs_weights,
        attrs_weights=attrs_weights,
        rng=rng,
    )


def get_objs_attrs_permutation_strategy_mixed(
    n_objs: int,
    n_attrs: int,
    objs_weights: Optional[Union[int, float, np.ndarray]] = None,
    attrs_weights: Optional[Union[int, float, np.ndarray]] = None,
    rng: rght.Seed = None,
) -> np.ndarray:
    weights = np.concatenate(
        (
            prepare_weights(
                objs_weights,
                n_objs,
                expand_none=True,
                normalize=False,
            ),
            prepare_weights(
                attrs_weights,
                n_attrs,
                expand_none=True,
                normalize=False,
            ),
        )
    )
    proba = prepare_weights(weights, n_objs + n_attrs)
    result = get_permutation(0, n_objs + n_attrs, proba, seed=rng)
    return result


ObjsAttrsPermutationStrategy = Literal[
    "attrs_before",
    "mixed",
    "objs_before",
]


OBJS_ATTRS_PERMUTATION_STRATEGIES: Mapping[
    ObjsAttrsPermutationStrategy,
    rght.ObjsAttrsPermutationStrategyFunction,
] = {
    "attrs_before": get_objs_attrs_permutation_strategy_attrs_before,
    "mixed": get_objs_attrs_permutation_strategy_mixed,
    "objs_before": get_objs_attrs_permutation_strategy_objs_before,
}


def get_objs_attrs_permutation(
    n_objs: int,
    n_attrs: int,
    objs_weights: Optional[Union[int, float, np.ndarray]] = None,
    attrs_weights: Optional[Union[int, float, np.ndarray]] = None,
    strategy: ObjsAttrsPermutationStrategy = "mixed",
    seed: rght.Seed = None,
) -> np.ndarray:
    if strategy not in get_args(ObjsAttrsPermutationStrategy):
        raise ValueError("Unrecognized permutation strategy")

    if n_objs < 0:
        raise ValueError("`n_objs` cannot be less than zero")
    if n_attrs < 0:
        raise ValueError("`n_attrs` cannot be less than zero")

    rng = np.random.default_rng(seed)

    result = OBJS_ATTRS_PERMUTATION_STRATEGIES[strategy](
        n_objs=n_objs,
        n_attrs=n_attrs,
        objs_weights=objs_weights,
        attrs_weights=attrs_weights,
        rng=rng,
    )

    return result


def get_objs_permutation(
    n_objs: int,
    objs_weights: Optional[Union[int, float, np.ndarray]] = None,
    seed: rght.Seed = None,
) -> np.ndarray:
    return get_objs_attrs_permutation(
        n_objs=n_objs,
        n_attrs=0,
        objs_weights=objs_weights,
        strategy="objs_before",
        seed=seed,
    )


def get_attrs_permutation(
    n_attrs: int,
    attrs_weights: Optional[Union[int, float, np.ndarray]] = None,
    seed: rght.Seed = None,
) -> np.ndarray:
    return get_objs_attrs_permutation(
        n_objs=0,
        n_attrs=n_attrs,
        attrs_weights=attrs_weights,
        strategy="attrs_before",
        seed=seed,
    )
