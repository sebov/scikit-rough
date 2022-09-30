"""Functions related to weights."""

from typing import Literal, Optional, Union, overload

import numpy as np


def _ensure_non_zero(values: np.ndarray) -> np.ndarray:
    """Ensure positive values for non negative array.

    For an input array that contains non-negative values ensure all output values are
    positive. If there are already no zeros in the input array then the original array
    is returned. Otherwise, :func:`numpy.nextafter` towards :obj:`numpy.inf` is used on
    all input values, i.e., also non zero inputs are the subject of the
    :func:`numpy.nextafter` function.

    The function assumes but does not check (for performance reasons) if the input array
    consists of only non-negative values.

    Args:
        values: Input array of non-negative values.

    Returns:
        An array of positive only (non-zero) values.
    """
    if (values == 0).any():
        values = np.nextafter(values, np.inf)
    return values


def normalize_weights(
    weights: np.ndarray,
) -> np.ndarray:
    """Normalize weights.

    Normalize input ``weights`` using 1-norm (manhattan) norm. The function is intended
    to be used for normalization of weights of elements under consideration (e.g.,
    attributes, objects/instances), thus preparing discrete probability distribution
    used later in various draw tasks. Some of the draw methods cannot handle 0-valued
    probabilities and therefore the ``normalize_weights`` function uses a special
    procedure when 0-valued elements are found in the input ``weights`` vector. In such
    a case :func:`numpy.nextafter` is used internally to increase all values towards
    :obj:`numpy.inf` before and after (to overcome edge cases with close to zero values)
    normalization.

    The function does not check for negative values in the input ``weights``. Therefore,
    using the function with such inputs may produce unexpected results, especially when
    the output of the function is later used as a discrete probability distribution.

    Args:
        weights: Values to be normalized.

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
    # to overcome edge cases - ensure that there are no zeros before normalization
    values = _ensure_non_zero(values)
    norm = np.linalg.norm(values, ord=1)
    if norm > 0:
        values = values / norm
    # but also after, as some of the resulting values (because of close to zero
    # numerical values) could have turned into zeros after normalization
    values = _ensure_non_zero(values)
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

    * ``weights`` can be :obj:`None`, then if
        * ``expand_none == True`` - uniform output of ``1`` repeated ``size`` times is
          produced
        * ``expand_none == False`` - :obj:`None` output is produced
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
            code:`weights is None and expand_none == False`.
        expand_none: Whether :obj:`None` weights input should be expanded to an array of
            non-null values. Defaults to True.
        normalize: Whether to normalize the output values. Defaults to :obj:`True`.

    Raises:
        ValueError: If :code:`size is None` or less than zero but it is necessary for
            producing the result. E.g., ``weights`` is one of :obj:`int` or :obj:`float`
            or :code:`weights is None` and :code:`expand_none == True`.

    Returns:
        Output weights.
    """
    if weights is None:
        if expand_none:
            weights = 1
        else:
            return None

    if isinstance(weights, (int, float)):
        if size is None:
            raise ValueError("`size` cannot be `None` for the specified `weights`")
        if size < 0:
            raise ValueError(
                "`size` cannot be less than zero for the specified `weights`"
            )
        weights = np.repeat(weights, size)

    if normalize:
        weights = normalize_weights(weights)

    return weights
