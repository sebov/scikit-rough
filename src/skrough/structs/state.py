from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

import skrough.typing as rght
from skrough.structs.group_index import GroupIndex

StateConfig = Mapping[str, Any]


ProcessingFunction = Callable[["ProcessingState"], Any]


@dataclass
class ProcessingState:  # pylint: disable=too-many-public-methods,too-many-instance-attributes
    processing_fun: ProcessingFunction | None
    rng: np.random.Generator | None
    config: StateConfig = field(default_factory=dict)

    _input_data_x: np.ndarray | None = None
    _input_data_x_counts: np.ndarray | None = None
    _input_data_y: np.ndarray | None = None
    _input_data_y_count: int | None = None

    _config_disorder_fun: rght.DisorderMeasure | None = None
    _config_consecutive_empty_iterations_max_count: int | None = None
    _config_daar_allowed_randomness: float | None = None
    _config_daar_fast: bool | None = None
    _config_daar_probes_count: int | None = None
    _config_daar_smoothing_parameter: float | None = None
    _config_epsilon: float | None = None
    _config_select_attrs_disorder_score_based_max_count: int | None = None
    _config_candidates_select_random_max_count: int | None = None
    _config_result_attrs_max_count: int | None = None
    _config_set_approx_threshold_to_current: bool | None = None

    _values_group_index: GroupIndex | None = None
    _values_x: np.ndarray | None = None
    _values_x_counts: np.ndarray | None = None
    _values_y: np.ndarray | None = None
    _values_y_count: int | None = None
    _values_result_objs: list[int] | None = None
    _values_result_attrs: list[int] | None = None
    _values_disorder_score_approx_threshold: float | None = None
    _values_disorder_score_base: float | None = None
    _values_disorder_score_total: float | None = None
    _values_consecutive_empty_iterations_count: int | None = None

    def get_rng(self) -> np.random.Generator:
        if self.rng is None:
            raise ValueError("empty rng")
        return self.rng

    def set_rng(self, val: np.random.Generator):
        self.rng = val

    def is_set_rng(self) -> bool:
        return self.rng is not None

    def get_input_data_x(self) -> np.ndarray:
        if self._input_data_x is None:
            raise ValueError("empty input_data_x")
        return self._input_data_x

    def set_input_data_x(self, val: np.ndarray):
        self._input_data_x = val

    def get_input_data_x_counts(self) -> np.ndarray:
        if self._input_data_x_counts is None:
            raise ValueError("empty input_data_x_count")
        return self._input_data_x_counts

    def set_input_data_x_counts(self, val: np.ndarray):
        self._input_data_x_counts = val

    def get_input_data_y(self) -> np.ndarray:
        if self._input_data_y is None:
            raise ValueError("empty input_data_y")
        return self._input_data_y

    def set_input_data_y(self, val: np.ndarray):
        self._input_data_y = val

    def get_input_data_y_count(self) -> int:
        if self._input_data_y_count is None:
            raise ValueError("empty input_data_y_count")
        return self._input_data_y_count

    def set_input_data_y_count(self, val: int):
        self._input_data_y_count = val

    def get_config_disorder_fun(self) -> rght.DisorderMeasure:
        if self._config_disorder_fun is None:
            raise ValueError("empty config_disorder_fun")
        return self._config_disorder_fun

    def set_config_disorder_fun(self, val: rght.DisorderMeasure):
        self._config_disorder_fun = val

    def get_config_consecutive_empty_iterations_max_count(self) -> int:
        if self._config_consecutive_empty_iterations_max_count is None:
            raise ValueError("empty config_consecutive_empty_iterations_max_count")
        return self._config_consecutive_empty_iterations_max_count

    def set_config_consecutive_empty_iterations_max_count(self, val: int):
        self._config_consecutive_empty_iterations_max_count = val

    def get_config_daar_allowed_randomness(self) -> float:
        if self._config_daar_allowed_randomness is None:
            raise ValueError("empty config_daar_allowed_randomness")
        return self._config_daar_allowed_randomness

    def set_config_daar_allowed_randomness(self, val: float):
        self._config_daar_allowed_randomness = val

    def get_config_daar_fast(self) -> bool:
        if self._config_daar_fast is None:
            raise ValueError("empty config_daar_fast")
        return self._config_daar_fast

    def set_config_daar_fast(self, val: bool):
        self._config_daar_fast = val

    def get_config_daar_probes_count(self) -> int:
        if self._config_daar_probes_count is None:
            raise ValueError("empty config_daar_probes_count")
        return self._config_daar_probes_count

    def set_config_daar_probes_count(self, val: int):
        self._config_daar_probes_count = val

    def get_config_daar_smoothing_parameter(self) -> float:
        if self._config_daar_smoothing_parameter is None:
            raise ValueError("empty config_daar_smoothing_parameter")
        return self._config_daar_smoothing_parameter

    def set_config_daar_smoothing_parameter(self, val: float):
        self._config_daar_smoothing_parameter = val

    def get_config_epsilon(self) -> float:
        if self._config_epsilon is None:
            raise ValueError("empty config_epsilon")
        return self._config_epsilon

    def set_config_epsilon(self, val: float):
        self._config_epsilon = val

    def get_config_select_attrs_disorder_score_based_max_count(self) -> int:
        if self._config_select_attrs_disorder_score_based_max_count is None:
            raise ValueError("empty config_select_attrs_disorder_score_based_max_count")
        return self._config_select_attrs_disorder_score_based_max_count

    def set_config_select_attrs_disorder_score_based_max_count(self, val: int):
        self._config_select_attrs_disorder_score_based_max_count = val

    def get_config_candidates_select_random_max_count(self) -> int:
        if self._config_candidates_select_random_max_count is None:
            raise ValueError("empty config_candidates_select_random_max_count")
        return self._config_candidates_select_random_max_count

    def set_config_candidates_select_random_max_count(self, val: int):
        self._config_candidates_select_random_max_count = val

    def is_set_config_candidates_select_random_max_count(self) -> bool:
        return self._config_candidates_select_random_max_count is not None

    def get_config_result_attrs_max_count(self) -> int:
        if self._config_result_attrs_max_count is None:
            raise ValueError("empty config_result_attrs_max_count")
        return self._config_result_attrs_max_count

    def set_config_result_attrs_max_count(self, val: int):
        self._config_result_attrs_max_count = val

    def is_set_config_result_attrs_max_count(self) -> bool:
        return self._config_result_attrs_max_count is not None

    def get_config_set_approx_threshold_to_current(self) -> bool:
        if self._config_set_approx_threshold_to_current is None:
            raise ValueError("empty config_set_approx_threshold_to_current")
        return self._config_set_approx_threshold_to_current

    def set_config_set_approx_threshold_to_current(self, val: bool):
        self._config_set_approx_threshold_to_current = val

    def get_values_group_index(self) -> GroupIndex:
        if self._values_group_index is None:
            raise ValueError("empty group_index")
        return self._values_group_index

    def set_values_group_index(self, val: GroupIndex):
        self._values_group_index = val

    def is_set_values_group_index(self) -> bool:
        return self._values_group_index is not None

    def get_values_x(self) -> np.ndarray:
        if self._values_x is None:
            raise ValueError("empty values_x")
        return self._values_x

    def set_values_x(self, val: np.ndarray):
        self._values_x = val

    def is_set_values_x(self) -> bool:
        return self._values_x is not None

    def get_values_x_counts(self) -> np.ndarray:
        if self._values_x_counts is None:
            raise ValueError("empty values_x_count")
        return self._values_x_counts

    def set_values_x_counts(self, val: np.ndarray):
        self._values_x_counts = val

    def is_set_values_x_counts(self) -> bool:
        return self._values_x_counts is not None

    def get_values_y(self) -> np.ndarray:
        if self._values_y is None:
            raise ValueError("empty values_y")
        return self._values_y

    def set_values_y(self, val: np.ndarray):
        self._values_y = val

    def get_values_y_count(self) -> int:
        if self._values_y_count is None:
            raise ValueError("empty values_y_count")
        return self._values_y_count

    def set_values_y_count(self, val: int):
        self._values_y_count = val

    def get_values_result_objs(self) -> list[int]:
        if self._values_result_objs is None:
            raise ValueError("empty values_result_objs")
        return self._values_result_objs

    def set_values_result_objs(self, val: list[int]):
        self._values_result_objs = val

    def is_set_values_result_objs(self) -> bool:
        return self._values_result_objs is not None

    def get_values_result_attrs(self) -> list[int]:
        if self._values_result_attrs is None:
            raise ValueError("empty values_result_attrs")
        return self._values_result_attrs

    def set_values_result_attrs(self, val: list[int]):
        self._values_result_attrs = val

    def is_set_values_result_attrs(self) -> bool:
        return self._values_result_attrs is not None

    def get_values_disorder_score_approx_threshold(self) -> float:
        if self._values_disorder_score_approx_threshold is None:
            raise ValueError("empty values_disorder_score_approx_threshold")
        return self._values_disorder_score_approx_threshold

    def set_values_disorder_score_approx_threshold(self, val: float):
        self._values_disorder_score_approx_threshold = val

    def get_values_disorder_score_base(self) -> float:
        if self._values_disorder_score_base is None:
            raise ValueError("empty values_disorder_score_base")
        return self._values_disorder_score_base

    def set_values_disorder_score_base(self, val: float):
        self._values_disorder_score_base = val

    def get_values_disorder_score_total(self) -> float:
        if self._values_disorder_score_total is None:
            raise ValueError("empty values_disorder_score_total")
        return self._values_disorder_score_total

    def set_values_disorder_score_total(self, val: float):
        self._values_disorder_score_total = val

    def get_values_consecutive_empty_iterations_count(self) -> int:
        if self._values_consecutive_empty_iterations_count is None:
            raise ValueError("empty values_consecutive_empty_iterations_count")
        return self._values_consecutive_empty_iterations_count

    def set_values_consecutive_empty_iterations_count(self, val: int):
        self._values_consecutive_empty_iterations_count = val

    def is_set_values_consecutive_empty_iterations_count(self) -> bool:
        return self._values_consecutive_empty_iterations_count is not None

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
