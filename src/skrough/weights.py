from typing import Literal, Optional, Union, cast, overload

import numpy as np
import numpy.typing as npt


def normalize_weights(weights: npt.ArrayLike) -> np.ndarray:
    """Normalize weights.

    Normalize weights to a sum equal to 1. The result will not contain zero elements
    even if there are zeros in the input. In such a case a small positive value
    (``np.finfo.eps``) is added to all input elements before normalization.

    Args:
        weights: Values to be normalized.

    Returns:
        Normalized weights.
    """
    values = np.asarray(weights, dtype=float)
    if any(values == 0):
        values += np.finfo(dtype=np.float64).eps
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

    Process weights into array form. Input weights can be given as a single value or an
    array of values. The following cases are handled in the function:

    * ``weights`` can be ``None``
        * ``expand_none == True`` - uniform output of ``n`` 1s is produced
        * ``expand_none == False`` - None output is produced
    * ``weights`` can be ``int`` or ``float`` - uniform output of ``n`` times repeated
        value of input ``weights`` is produced
    * ``weights`` can be ``np.ndarray`` - input ``weights`` are taken as is

    Additional normalization step (using ``normalize_weights`` function) is performed
    for the above result when ``normalize == True``.

    Args:
        weights: Value(s) to be processed.
        n: Output length. May be omitted if
            ``weights is None and expand_none == False``.
        expand_none: Whether ``None`` weights input should be expanded to an array of
            non-null values. Defaults to True.
        normalize: Whether to normalize the output values. Defaults to True.

    Returns:
        Output weights.
    """
    if not isinstance(weights, np.ndarray) and size is None:
        raise ValueError("``n`` argument cannot be None for the specified ``weights``")

    if weights is None:
        if expand_none:
            weights = 1
        else:
            return None
    if isinstance(weights, (int, float)):
        size = cast(int, size)
        weights = np.repeat(weights, size)
    if normalize:
        weights = normalize_weights(weights)
    return weights
