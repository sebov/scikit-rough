import abc
from typing import Any, Protocol

import numpy as np

import skrough.typing as rght
from skrough.structs.description_node import DescriptionNode
from skrough.structs.state import ProcessingState


class PredictStrategyFunction(Protocol):
    @abc.abstractmethod
    def __call__(
        self,
        reference_ids: np.ndarray,
        reference_data_y: np.ndarray,
        predict_ids: np.ndarray,
        seed: rght.Seed = None,
    ) -> Any:
        raise NotImplementedError


# no-answer strategy - what should be the answer when a classifier "do not know"
class NoAnswerStrategyFunction(Protocol):
    @abc.abstractmethod
    def __call__(
        self,
        reference_data_y: np.ndarray,
        seed: rght.Seed = None,
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
        rng: rght.Seed = None,
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
        elements: rght.Elements,
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
    ) -> rght.Elements:
        raise NotImplementedError


class ProcessElementsHook(Protocol):
    @staticmethod
    @abc.abstractmethod
    def __call__(
        state: ProcessingState,
        elements: rght.Elements,
    ) -> rght.Elements:
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
