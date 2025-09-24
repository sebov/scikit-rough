from typing import Any, Callable, Mapping, MutableMapping

import numpy as np
from dataclasses import dataclass, field

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
