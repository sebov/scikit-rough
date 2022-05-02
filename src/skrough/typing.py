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

StopHook = Callable[
    [ProcessingState],
    bool,
]

InnerStopHook = Callable[
    [ProcessingState, Elements],
    bool,
]

UpdateStateHook = Callable[
    [ProcessingState],
    None,
]

ProduceElementsHook = Callable[
    [ProcessingState],
    Elements,
]

ProcessElementsHook = Callable[
    [ProcessingState, Elements],
    Elements,
]

PrepareResultHook = Callable[
    [ProcessingState],
    Any,
]


T = TypeVar("T")


OptionalOneOrSequence = Optional[
    Union[
        T,
        Sequence[T],
    ]
]


OneOrSequence = Union[
    T,
    Sequence[T],
]


class StopFunction(Protocol):
    @staticmethod
    def __call__(
        state: ProcessingState,
        raise_exception: bool = True,
    ) -> float:
        ...


class InnerStopFunction(Protocol):
    @staticmethod
    def __call__(
        state: ProcessingState,
        elements: Elements,
        raise_exception: bool = True,
    ) -> float:
        ...


UpdateStateFunction = Callable[[ProcessingState], None]
ProduceElementsFunction = Callable[[ProcessingState], Elements]
ProcessElementsFunction = Callable[[ProcessingState, Elements], Elements]
