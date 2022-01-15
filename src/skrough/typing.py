import typing

import numpy as np

dataType = np.ndarray
dataCountsType = np.ndarray
decDataType = np.ndarray
decDataCountType = int

valuesType = np.ndarray
valuesCountType = int

groupIndexType = np.ndarray
groupIndexCountType = int

distributionType = np.ndarray
distributionCountType = int
chaosMeasureFunType = typing.Callable[[distributionType, distributionCountType], float]

randomStateType = typing.Union[None, int, np.random.RandomState]


AttrsType = list[int]
ObjectsType = list[int]
