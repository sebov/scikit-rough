"""Typing module."""

import abc
import itertools
from typing import Any, Callable, List, Optional, Protocol, Sequence, TypeVar, Union

import numpy as np
import numpy.typing as npt

from skrough.structs.description_node import DescriptionNode
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
Locations = npt.NDArray[np.int64]
LocationsLike = Union[Sequence[int], Locations]

T = TypeVar("T")
OneOrSequence = Union[
    T,
    Sequence[T],
]


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
        objs_weights: Optional[Union[int, float, np.ndarray]] = None,
        attrs_weights: Optional[Union[int, float, np.ndarray]] = None,
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

    @abc.abstractmethod
    def get_config_keys(self) -> List[str]:
        """Get a list of "config" keys used by the instance and its descendants.

        Returns:
            A list of "config" keys used by the instance and its descendants.
        """

    @abc.abstractmethod
    def get_input_data_keys(self) -> List[str]:
        """Get a list of "input" keys used by the instance and its descendants.

        Returns:
            A list of "input" keys used by the instance and its descendants.
        """

    @abc.abstractmethod
    def get_values_keys(self) -> List[str]:
        """Get a list of "values" keys used by the instance and its descendants.

        Returns:
            A list of "values" keys used by the instance and its descendants.
        """

    @staticmethod
    def _get_keys_from_elements(
        children: Sequence,
        inspect_keys_function: Callable,
    ) -> List[str]:
        return list(
            set(
                itertools.chain.from_iterable(
                    [inspect_keys_function(child) for child in children],
                )
            )
        )
