from typing import Any, Callable, Mapping, MutableMapping, Optional, Union

import numpy as np

from skrough.structs.reduct import (  # noqa: F401 # pylint: disable=unused-import
    ReductLike,
)

ChaosMeasureReturnType = float
ChaosMeasure = Callable[[np.ndarray, int], ChaosMeasureReturnType]

Seed = Optional[Union[int, np.random.SeedSequence, np.random.Generator]]


Config = Mapping[str, Any]
StateValues = MutableMapping[str, Any]

InitStateHook = Callable[[np.ndarray, np.ndarray, np.ndarray, int], None]
