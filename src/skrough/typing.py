from typing import Callable, Optional, Union

import numpy as np

ChaosMeasure = Callable[[np.ndarray, int], float]

RandomState = Optional[Union[int, np.random.SeedSequence, np.random.Generator]]
