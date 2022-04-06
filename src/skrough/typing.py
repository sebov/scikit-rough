from typing import Any, Callable, Optional, Union

import numpy as np

from skrough.structs.attrs_subset import (  # noqa: F401 # pylint: disable=unused-import
    AttrsSubsetLike,
)
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import GrowShrinkState

ChaosMeasureReturnType = float
ChaosMeasure = Callable[[np.ndarray, int], ChaosMeasureReturnType]

Seed = Optional[Union[int, np.random.SeedSequence, np.random.Generator]]

GSInitStateHook = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int, GrowShrinkState],
    None,
]

GSGrowStopHook = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int, GrowShrinkState],
    bool,
]

GSGrowCandidateAttrsHook = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int, GrowShrinkState, np.ndarray],
    np.ndarray,
]

GSGrowSelectAttrsHook = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int, GrowShrinkState, np.ndarray],
    np.ndarray,
]

GSGrowPostSelectAttrsHook = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int, GrowShrinkState, np.ndarray],
    np.ndarray,
]

GSShrinkCandidateAttrsHook = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int, GrowShrinkState],
    np.ndarray,
]

GSShrinkAcceptGroupIndexHook = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int, GrowShrinkState, GroupIndex],
    bool,
]

GSFinalizeStateHook = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int, GrowShrinkState],
    None,
]

GSPrepareResultHook = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int, GrowShrinkState],
    Any,
]
