"""Typing module."""

from typing import Callable, Sequence

import numpy as np
import numpy.typing as npt

# Disorder measures
DisorderMeasureReturnType = float
# """Return type of disorder measure functions."""
DisorderMeasure = Callable[[np.ndarray, int], DisorderMeasureReturnType]
# """A type/signature of disorder measure functions."""


# Random
Seed = int | np.int64 | np.random.SeedSequence | np.random.Generator | None
# """A type for values which can be used as a random seed."""


# Collections
Elements = Sequence | np.ndarray
IndexList = npt.NDArray[np.int64]
IndexListLike = Sequence[int] | IndexList
