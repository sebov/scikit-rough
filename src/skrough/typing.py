from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from skrough.structs.attrs_subset import (  # noqa: F401 # pylint: disable=unused-import
    AttrsSubsetLike,
)
from skrough.structs.group_index import GroupIndex
from skrough.structs.state import GrowShrinkState

ChaosMeasureReturnType = float
ChaosMeasure = Callable[[np.ndarray, int], ChaosMeasureReturnType]

Seed = Optional[Union[int, np.random.SeedSequence, np.random.Generator]]

GSUpdateStateHook = Callable[
    [GrowShrinkState],
    None,
]

GSGrowStopHook = Callable[
    [GrowShrinkState],
    bool,
]

GSElements = Union[Sequence, np.ndarray]

GSGrowPreCandidatesHook = Callable[
    [GrowShrinkState],
    GSElements,
]

GSGrowCandidatesHook = Callable[
    [GrowShrinkState, GSElements],
    GSElements,
]

GSGrowSelectAttrsHook = Callable[
    [GrowShrinkState, GSElements],
    GSElements,
]

GSGrowPostSelectHook = Callable[
    [GrowShrinkState, GSElements],
    GSElements,
]

GSShrinkCandidatesHook = Callable[
    [GrowShrinkState],
    GSElements,
]

GSShrinkAcceptGroupIndexHook = Callable[
    [GrowShrinkState, GroupIndex],
    bool,
]

GSPrepareResultHook = Callable[
    [GrowShrinkState],
    Any,
]
