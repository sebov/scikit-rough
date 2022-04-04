from typing import Any, Callable, Optional, Union

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

GSCheckStopHook = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int, GrowShrinkState],
    bool,
]

GSGetCandidateAttrsHook = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int, GrowShrinkState, np.ndarray],
    np.ndarray,
]

GSSelectAttrsHook = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int, GrowShrinkState, np.ndarray],
    np.ndarray,
]

GSPrepareResultHook = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int, GrowShrinkState],
    Any,
]
