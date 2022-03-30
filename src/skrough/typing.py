from typing import Callable, Optional, Union

import numpy as np

from skrough.containers import ReductLike  # noqa: F401 # pylint: disable=unused-import

ChaosMeasureReturnType = float
ChaosMeasure = Callable[[np.ndarray, int], ChaosMeasureReturnType]

Seed = Optional[Union[int, np.random.SeedSequence, np.random.Generator]]
