from typing import Literal, Optional, Union, overload

import numpy as np


def normalize_weights(
    weights: np.ndarray,
    eps: np.float64 = np.finfo(dtype=np.float64).eps,
) -> np.ndarray:
    """Normalize weights.

    Normalize input ``weights`` using 1-norm (manhattan) norm. The function is intended
    to be used for normalization of weights of elements under consideration (e.g.,
    attributes, objects/instances), thus preparing discrete probability distribution
    used later in various draw tasks. Some of the draw methods cannot handle 0-valued
    probabilities and therefore the ``normalize_weights`` function uses a special
    procedure when 0-valued elements are found in the input ``weights`` vector. In such
    a case ``eps`` value is added to each ``weights`` elements before normalization.

    The function does not check for negative values in the input ``weights``. Therefore,
    using the function with such inputs may produce unexpected results, especially when
    the output of the function is later used as a discrete probability distribution.

    Args:
        weights: Values to be normalized.
        eps: Value to be added to all element before actual normalization if at least
            one ``0`` is found in the input ``weights``. Defaults to
            ``np.finfo(dtype=np.float64).eps``

    Returns:
        Normalized weights.

    Examples:
        >>> normalize_weights(np.asarray([1, 1, 2]))
        array([0.25, 0.25, 0.5])
        >>> normalize_weights(np.asarray([1, 3]))
        array([0.25, 0.75])
        >>> normalize_weights(np.asarray([0, 0]))
        array([0.5, 0.5])
        >>> normalize_weights(np.asarray([0, 1]))
        array([2.22044605e-16, 1.00000000e+00])
        >>> normalize_weights(np.asarray([-1, 1]))
        array([-0.5, 0.5])
    """
    values = np.asarray(weights, dtype=np.float64)
    if (values == 0).any():
        values += eps
    norm = np.linalg.norm(values, ord=1)
    if norm > 0:
        values = values / norm
    return values


@overload
def prepare_weights(
    weights: Optional[Union[int, float, np.ndarray]],
    size: int,
    *,
    expand_none: Literal[True] = True,
    normalize: bool = True,
) -> np.ndarray:
    ...


@overload
def prepare_weights(
    weights: Optional[Union[int, float, np.ndarray]],
    size: Optional[int] = None,
    *,
    expand_none: Literal[False],
    normalize: bool = True,
) -> Optional[np.ndarray]:
    ...


def prepare_weights(
    weights: Optional[Union[int, float, np.ndarray]],
    size: Optional[int] = None,
    *,
    expand_none: bool = True,
    normalize: bool = True,
) -> Optional[np.ndarray]:
    """Prepare weights.

    Process ``weights`` into an array form. The input ``weights`` can be given as a
    scalar value or an array-like structure of values. The following cases are handled
    in the function:

    * ``weights`` can be ``None``, then if
        * ``expand_none == True`` - uniform output of ``1`` repeated ``size`` times is
          produced
        * ``expand_none == False`` - ``None`` output is produced
    * ``weights`` can be ``int`` or ``float`` - uniform output of ``weights`` (scalar)
        value repeated ``size`` times is produced
    * ``weights`` can be ``np.ndarray`` - input ``weights`` are taken as is and in this
      case ``size`` parameter is ignored

    Additional normalization step (using :func:`normalize_weights` function) is
    performed for the above result when ``normalize == True``. All the remarks of
    :func:`normalize_weights` applies when negative values are present. In such a case
    the function will not produce a discrete probability distribution.

    Args:
        weights: Value(s) to be processed.
        size: Output length. May be omitted if
            ``weights is None and expand_none == False``.
        expand_none: Whether ``None`` weights input should be expanded to an array of
            non-null values. Defaults to True.
        normalize: Whether to normalize the output values. Defaults to True.

    Returns:
        Output weights.

    Raises:
        ValueError: If ``size`` is ``None`` but it is necessary for producing the
            result. E.g., ``weights`` is one of ``int`` or ``float`` or ``weights is
            None`` and ``expand_none == True``.
    """
    if weights is None:
        if expand_none:
            weights = 1
        else:
            return None

    if isinstance(weights, (int, float)):
        if size is None:
            raise ValueError("`size` cannot be None for the specified `weights`")
        weights = np.repeat(weights, size)

    if normalize:
        weights = normalize_weights(weights)

    return weights
