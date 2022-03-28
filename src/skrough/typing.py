from typing import Callable, List, Optional, Union

import numpy as np

import skrough as rgh

ReductLike = Union[rgh.containers.Reduct, List[int]]

ChaosMeasureReturnType = float
ChaosMeasure = Callable[[np.ndarray, int], ChaosMeasureReturnType]

Seed = Optional[Union[int, np.random.SeedSequence, np.random.Generator]]
