from typing import Any, Callable, Optional, Protocol, Sequence, TypeVar, Union

import numpy as np

from skrough.structs.attrs_subset import (  # noqa: F401 # pylint: disable=unused-import
    AttrsSubsetLike,
)
from skrough.structs.state import ProcessingState

ChaosMeasureReturnType = float
ChaosMeasure = Callable[[np.ndarray, int], ChaosMeasureReturnType]

Seed = Optional[Union[int, np.random.SeedSequence, np.random.Generator]]

Elements = Union[Sequence, np.ndarray]


# Function collection types

T = TypeVar("T")

OneOrSequence = Union[
    T,
    Sequence[T],
]


# Processing/stage functions


class StopFunction(Protocol):
    @staticmethod
    def __call__(
        state: ProcessingState,
        raise_exception: bool = True,
    ) -> bool:
        ...


class InnerStopFunction(Protocol):
    @staticmethod
    def __call__(
        state: ProcessingState,
        elements: Elements,
        raise_exception: bool = True,
    ) -> bool:
        ...


UpdateStateFunction = Callable[[ProcessingState], None]
ProduceElementsFunction = Callable[[ProcessingState], Elements]
ProcessElementsFunction = Callable[[ProcessingState, Elements], Elements]
PrepareResultFunction = Callable[[ProcessingState], Any]


# Hook functions - to be composed/aggregated into processing/stage functions
StopHook = Callable[[ProcessingState], bool]
InnerStopHook = Callable[[ProcessingState, Elements], bool]
UpdateStateHook = UpdateStateFunction
ProduceElementsHook = ProduceElementsFunction
ProcessElementsHook = ProcessElementsFunction
