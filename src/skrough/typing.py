from typing import Callable, Optional, Union

import numpy as np

from skrough.structs.reduct import (  # noqa: F401 # pylint: disable=unused-import
    ReductLike,
)
from skrough.structs.state import GrowShrinkState

ChaosMeasureReturnType = float
ChaosMeasure = Callable[[np.ndarray, int], ChaosMeasureReturnType]

Seed = Optional[Union[int, np.random.SeedSequence, np.random.Generator]]

GSInitStateHook = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int, GrowShrinkState],
    None,
]

GSStopHook = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int, GrowShrinkState],
    bool,
]
