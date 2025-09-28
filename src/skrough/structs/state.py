from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from skrough.structs.group_index import GroupIndex

StateConfig = Mapping[str, Any]


ProcessingFunction = Callable[["ProcessingState"], Any]


@dataclass
class ProcessingState:  # pylint: disable=too-many-public-methods,too-many-instance-attributes
    processing_fun: ProcessingFunction | None
    rng: np.random.Generator | None
    config: StateConfig = field(default_factory=dict)

    _group_index: GroupIndex | None = None
    _input_data_x: np.ndarray | None = None
    _input_data_x_counts: np.ndarray | None = None
    _input_data_y: np.ndarray | None = None
    _input_data_y_count: int | None = None
    _values_x: np.ndarray | None = None
    _values_x_counts: np.ndarray | None = None
    _values_y: np.ndarray | None = None
    _values_y_count: int | None = None
    _values_result_objs: list[int] | None = None
    _values_result_attrs: list[int] | None = None
    _values_disorder_score_approx_threshold: float | None = None
    _values_disorder_score_base: float | None = None
    _values_disorder_score_total: float | None = None

    _values_consecutive_empty_iterations_count: int = 0

    def set_rng(self, val: np.random.Generator):
        self.rng = val

    def get_rng(self) -> np.random.Generator:
        if self.rng is None:
            raise ValueError("empty rng")
        return self.rng

    def is_set_rng(self) -> bool:
        return self.rng is not None

    def set_group_index(self, val: GroupIndex):
        self._group_index = val

    def get_group_index(self) -> GroupIndex:
        if self._group_index is None:
            raise ValueError("empty group_index")
        return self._group_index

    def is_set_group_index(self) -> bool:
        return self._group_index is not None

    def set_input_data_x(self, val: np.ndarray):
        self._input_data_x = val

    def get_input_data_x(self) -> np.ndarray:
        if self._input_data_x is None:
            raise ValueError("empty input_data_x")
        return self._input_data_x

    def set_input_data_x_counts(self, val: np.ndarray):
        self._input_data_x_counts = val

    def get_input_data_x_counts(self) -> np.ndarray:
        if self._input_data_x_counts is None:
            raise ValueError("empty input_data_x_count")
        return self._input_data_x_counts

    def set_input_data_y(self, val: np.ndarray):
        self._input_data_y = val

    def get_input_data_y(self) -> np.ndarray:
        if self._input_data_y is None:
            raise ValueError("empty input_data_y")
        return self._input_data_y

    def set_input_data_y_count(self, val: int):
        self._input_data_y_count = val

    def get_input_data_y_count(self) -> int:
        if self._input_data_y_count is None:
            raise ValueError("empty input_data_y_count")
        return self._input_data_y_count

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

    def set_values_result_attrs(self, val: list[int]):
        self._values_result_attrs = val

    def get_values_result_attrs(self) -> list[int]:
        if self._values_result_attrs is None:
            raise ValueError("empty values_result_attrs")
        return self._values_result_attrs

    def is_set_values_result_attrs(self) -> bool:
        return self._values_result_attrs is not None

    def set_values_disorder_score_approx_threshold(self, val: float):
        self._values_disorder_score_approx_threshold = val

    def get_values_disorder_score_approx_threshold(self) -> float:
        if self._values_disorder_score_approx_threshold is None:
            raise ValueError("empty values_disorder_score_approx_threshold")
        return self._values_disorder_score_approx_threshold

    def set_values_disorder_score_base(self, val: float):
        self._values_disorder_score_base = val

    def get_values_disorder_score_base(self) -> float:
        if self._values_disorder_score_base is None:
            raise ValueError("empty values_disorder_score_base")
        return self._values_disorder_score_base

    def set_values_disorder_score_total(self, val: float):
        self._values_disorder_score_total = val

    def get_values_disorder_score_total(self) -> float:
        if self._values_disorder_score_total is None:
            raise ValueError("empty values_disorder_score_total")
        return self._values_disorder_score_total

    def set_values_consecutive_empty_iterations_count(self, val: int):
        self._values_consecutive_empty_iterations_count = val

    def get_values_consecutive_empty_iterations_count(self) -> int:
        return self._values_consecutive_empty_iterations_count

    @classmethod
    def from_optional(
        cls,
        processing_fun: ProcessingFunction | None,
        rng: np.random.Generator | None = None,
        config: StateConfig | None = None,
    ):
        optional_kwargs = {}
        if config is not None:
            optional_kwargs["config"] = config
        return cls(
            rng=rng,
            processing_fun=processing_fun,
            **optional_kwargs,
        )
