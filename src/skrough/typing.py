"""Typing module"""
from typing import Any, Callable, Optional, Protocol, Sequence, TypeVar, Union

import numpy as np
import numpy.typing as npt

from skrough.structs.state import ProcessingState

# Chaos measures
ChaosMeasureReturnType = float
# """Return type of chaos measure functions."""
ChaosMeasure = Callable[[np.ndarray, int], ChaosMeasureReturnType]
# """A type/signature of chaos measure functions."""


# Random
Seed = Optional[Union[int, np.random.SeedSequence, np.random.Generator]]
# """A type for values which can be used as a random seed."""


# Collections
Elements = Union[Sequence, np.ndarray]
Objs = npt.NDArray[np.int64]
ObjsLike = Union[Sequence[int], Objs]
Attrs = npt.NDArray[np.int64]
AttrsLike = Union[Sequence[int], Attrs]

T = TypeVar("T")
OneOrSequence = Union[
    T,
    Sequence[T],
]


# Predict strategy
class PredictStrategyFunction(Protocol):
    @staticmethod
    def __call__(
        reference_ids: np.ndarray,
        reference_y: np.ndarray,
        predict_ids: np.ndarray,
        seed: Seed,
    ) -> Any:
        ...


# Permutation strategy
class ObjsAttrsPermutationStrategyFunction(Protocol):
    @staticmethod
    def __call__(
        n_objs: int,
        n_attrs: int,
        objs_weights: Optional[Union[int, float, np.ndarray]] = None,
        attrs_weights: Optional[Union[int, float, np.ndarray]] = None,
        rng: Seed = None,
    ) -> Any:
        ...


# Processing/stage functions
class StopFunction(Protocol):
    @staticmethod
    def __call__(
        state: ProcessingState,
        raise_exception: bool,
    ) -> bool:
        ...


class InnerStopFunction(Protocol):
    @staticmethod
    def __call__(
        state: ProcessingState,
        elements: Elements,
        raise_exception: bool,
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
