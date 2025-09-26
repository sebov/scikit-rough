from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping

import numpy as np

from skrough.structs.group_index import GroupIndex

StateConfig = Mapping[str, Any]
StateInputData = Mapping[str, Any]
StateValues = MutableMapping[str, Any]


ProcessingFunction = Callable[["ProcessingState"], Any]


@dataclass
class ProcessingState:
    rng: np.random.Generator
    processing_fun: ProcessingFunction | None
    config: StateConfig = field(default_factory=dict)
    input_data: StateInputData = field(default_factory=dict)
    values: StateValues = field(default_factory=dict)

    _group_index: GroupIndex | None = None
    _values_x: np.ndarray | None = None
    _values_x_counts: np.ndarray | None = None
    _values_y: np.ndarray | None = None
    _values_y_count: int | None = None
    _values_result_objs: list[int] | None = None

    def set_group_index(self, val: GroupIndex):
        self._group_index = val

    def get_group_index(self) -> GroupIndex:
        if self._group_index is None:
            raise ValueError("empty group_index")
        return self._group_index

    def is_set_group_index(self) -> bool:
        return self._group_index is not None

    def set_values_x(self, val: np.ndarray):
        self._values_x = val

    def get_values_x(self) -> np.ndarray:
        if self._values_x is None:
            raise ValueError("empty values_x")
        return self._values_x

    def is_set_values_x(self) -> bool:
        return self._values_x is not None

    def set_values_x_counts(self, val: np.ndarray):
        self._values_x_counts = val

    def get_values_x_counts(self) -> np.ndarray:
        if self._values_x_counts is None:
            raise ValueError("empty values_x_count")
        return self._values_x_counts

    def is_set_values_x_counts(self) -> bool:
        return self._values_x_counts is not None

    def set_values_y(self, val: np.ndarray):
        self._values_y = val

    def get_values_y(self) -> np.ndarray:
        if self._values_y is None:
            raise ValueError("empty values_y")
        return self._values_y

    def set_values_y_count(self, val: int):
        self._values_y_count = val

    def get_values_y_count(self) -> int:
        if self._values_y_count is None:
            raise ValueError("empty values_y_count")
        return self._values_y_count

    def set_values_result_objs(self, val: list[int]):
        self._values_result_objs = val

    def get_values_result_objs(self) -> list[int]:
        if self._values_result_objs is None:
            raise ValueError("empty values_result_objs")
        return self._values_result_objs

    def is_set_values_result_objs(self) -> bool:
        return self._values_result_objs is not None

    @classmethod
    def from_optional(
        cls,
        rng: np.random.Generator,
        processing_fun: ProcessingFunction | None,
        config: StateConfig | None = None,
        input_data: StateInputData | None = None,
        values: StateValues | None = None,
    ):
        optional_kwargs = {}
        if config is not None:
            optional_kwargs["config"] = config
        if input_data is not None:
            optional_kwargs["input_data"] = input_data
        if values is not None:
            optional_kwargs["values"] = values
        return cls(
            rng=rng,
            processing_fun=processing_fun,
            **optional_kwargs,
        )
