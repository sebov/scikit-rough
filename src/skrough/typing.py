"""Typing module."""

import abc
from typing import Any, Callable, Protocol, Sequence

import numpy as np
import numpy.typing as npt

from skrough.structs.description_node import DescriptionNode
from skrough.structs.state import ProcessingState

# Disorder measures
DisorderMeasureReturnType = float
# """Return type of disorder measure functions."""
DisorderMeasure = Callable[[np.ndarray, int], DisorderMeasureReturnType]
# """A type/signature of disorder measure functions."""


# Random
Seed = int | np.random.SeedSequence | np.random.Generator | None
# """A type for values which can be used as a random seed."""


# Collections
Elements = Sequence | np.ndarray
IndexList = npt.NDArray[np.int64]
IndexListLike = Sequence[int] | IndexList


# Predict strategy
class PredictStrategyFunction(Protocol):
    @abc.abstractmethod
    def __call__(
        self,
        reference_ids: np.ndarray,
        reference_data_y: np.ndarray,
        predict_ids: np.ndarray,
        seed: Seed = None,
    ) -> Any:
        raise NotImplementedError


# no-answer strategy - what should be the answer when a classifier "do not know"
class NoAnswerStrategyFunction(Protocol):
    @abc.abstractmethod
    def __call__(
        self,
        reference_data_y: np.ndarray,
        seed: Seed = None,
    ) -> Any:
        raise NotImplementedError


# Permutation strategy
class ObjsAttrsPermutationStrategyFunction(Protocol):
    @staticmethod
    @abc.abstractmethod
    def __call__(
        n_objs: int,
        n_attrs: int,
        objs_weights: int | float | np.ndarray | None = None,
        attrs_weights: int | float | np.ndarray | None = None,
        rng: Seed = None,
    ) -> Any:
        raise NotImplementedError


# Processing/stage functions
class PrepareResultFunction(Protocol):
    @staticmethod
    @abc.abstractmethod
    def __call__(
        state: ProcessingState,
    ) -> Any:
        raise NotImplementedError


# Hook functions - to be composed/aggregated into processing/stage functions
class StopHook(Protocol):
    @staticmethod
    @abc.abstractmethod
    def __call__(
        state: ProcessingState,
    ) -> bool:
        raise NotImplementedError


class InnerStopHook(Protocol):
    @staticmethod
    @abc.abstractmethod
    def __call__(
        state: ProcessingState,
        elements: Elements,
    ) -> bool:
        raise NotImplementedError


class UpdateStateHook(Protocol):
    @staticmethod
    @abc.abstractmethod
    def __call__(
        state: ProcessingState,
    ) -> None:
        raise NotImplementedError


class ProduceElementsHook(Protocol):
    @staticmethod
    @abc.abstractmethod
    def __call__(
        state: ProcessingState,
    ) -> Elements:
        raise NotImplementedError


class ProcessElementsHook(Protocol):
    @staticmethod
    @abc.abstractmethod
    def __call__(
        state: ProcessingState,
        elements: Elements,
    ) -> Elements:
        raise NotImplementedError


# Describable
class Describable(abc.ABC):
    @abc.abstractmethod
    def get_description_graph(self) -> DescriptionNode:
        """Get a description graph.

        Prepare a description structure for the instance.

        Returns:
            A description graph structure representing the instance.
        """
