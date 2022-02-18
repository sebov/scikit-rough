import typing

import numpy as np

ChaosMeasureFunType = typing.Callable[[np.ndarray, np.int_], float]

RandomStateType = typing.Union[None, int, np.random.RandomState]
