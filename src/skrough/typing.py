from typing import Any, Callable, Optional, Protocol, Sequence, TypeVar, Union

import numpy as np

from skrough.structs.attrs_subset import (  # noqa: F401 # pylint: disable=unused-import
    AttrsSubsetLike,
)
from skrough.structs.state import GrowShrinkState

ChaosMeasureReturnType = float
ChaosMeasure = Callable[[np.ndarray, int], ChaosMeasureReturnType]

Seed = Optional[Union[int, np.random.SeedSequence, np.random.Generator]]

Elements = Union[Sequence, np.ndarray]

StopHook = Callable[
    [GrowShrinkState],
    bool,
]

InnerStopHook = Callable[
    [GrowShrinkState, Elements],
    bool,
]

UpdateStateHook = Callable[
    [GrowShrinkState],
    None,
]

ProduceElementsHook = Callable[
    [GrowShrinkState],
    Elements,
]

ProcessElementsHook = Callable[
    [GrowShrinkState, Elements],
    Elements,
]

PrepareResultHook = Callable[
    [GrowShrinkState],
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
        state: GrowShrinkState,
        raise_exception: bool = True,
    ) -> float:
        ...


class InnerStopFunction(Protocol):
    @staticmethod
    def __call__(
        state: GrowShrinkState,
        elements: Elements,
        raise_exception: bool = True,
    ) -> float:
        ...


UpdateStateFunction = Callable[[GrowShrinkState], None]
ProduceElementsFunction = Callable[[GrowShrinkState], Elements]
ProcessElementsFunction = Callable[[GrowShrinkState, Elements], Elements]
