import typing

import numpy as np

ChaosMeasure = typing.Callable[[np.ndarray, np.int_], float]

RandomState = typing.Union[None, int, np.random.RandomState]
