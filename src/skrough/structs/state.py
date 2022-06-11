from typing import Any, Callable, Mapping, MutableMapping, Optional

import numpy as np
from attrs import define, field

StateConfig = Mapping[str, Any]
StateInputData = Mapping[str, Any]
StateValues = MutableMapping[str, Any]


@define
class ProcessingState:
    rng: np.random.Generator
    processing_fun: Callable[["ProcessingState"], Any]
    config: StateConfig = field(factory=dict)
    input_data: StateInputData = field(factory=dict)
    values: StateValues = field(factory=dict)

    @classmethod
    def create_from_optional(
        cls,
        rng: np.random.Generator,
        processing_fun: Callable[["ProcessingState"], Any],
        config: Optional[StateConfig] = None,
        input_data: Optional[StateInputData] = None,
        values: Optional[StateValues] = None,
    ):
        optional_kwargs = {}
        if config is not None:
            optional_kwargs[cls.config.__name__] = config  # type: ignore
        if input_data is not None:
            optional_kwargs[cls.input_data.__name__] = input_data  # type: ignore
        if values is not None:
            optional_kwargs[cls.values.__name__] = values  # type: ignore
        return cls(
            rng=rng,
            processing_fun=processing_fun,
            **optional_kwargs,  # type: ignore
        )
