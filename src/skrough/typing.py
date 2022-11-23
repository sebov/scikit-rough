"""Typing module."""

import abc
import itertools
import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import numpy.typing as npt

from skrough.structs.description_node import DescriptionNode
from skrough.structs.state import ProcessingState

logger = logging.getLogger(__name__)


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
class PrepareResultFunction(Protocol):
    @staticmethod
    def __call__(
        state: ProcessingState,
    ) -> Any:
        ...


# Hook functions - to be composed/aggregated into processing/stage functions
class StopHook(Protocol):
    @staticmethod
    def __call__(
        state: ProcessingState,
    ) -> bool:
        ...


class InnerStopHook(Protocol):
    @staticmethod
    def __call__(
        state: ProcessingState,
        elements: Elements,
    ) -> bool:
        ...


class UpdateStateHook(Protocol):
    @staticmethod
    def __call__(
        state: ProcessingState,
    ) -> None:
        ...


class ProduceElementsHook(Protocol):
    @staticmethod
    def __call__(
        state: ProcessingState,
    ) -> Elements:
        ...


class ProcessElementsHook(Protocol):
    @staticmethod
    def __call__(
        state: ProcessingState,
        elements: Elements,
    ) -> Elements:
        ...


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
        determine_keys_function: Callable,
    ) -> List[str]:
        return list(
            set(
                itertools.chain.from_iterable(
                    [determine_keys_function(child) for child in children],
                )
            )
        )

    def check_compatibility(
        self,
        state: ProcessingState,
        verbose: bool = False,
    ) -> Union[bool, Tuple[bool, Dict[str, List[str]]]]:
        config_keys_ok = True
        input_data_keys_ok = True
        verbose_report = {}
        if not set(self.get_config_keys()).issubset(state.config.keys()):
            logger.info("some of the required config keys are not present in the state")
            config_keys_ok = False
            if verbose:
                verbose_report["missing_config_keys"] = list(
                    set(self.get_config_keys()).difference(state.config.keys())
                )
        if not set(self.get_input_data_keys()).issubset(state.input_data.keys()):
            logger.info(
                "some of the required input data keys are not present in the state"
            )
            input_data_keys_ok = False
            if verbose:
                verbose_report["missing_input_data_keys"] = list(
                    set(self.get_input_data_keys()).difference(state.input_data.keys())
                )

        result = config_keys_ok and input_data_keys_ok
        if verbose:
            return result, verbose_report
        return result
