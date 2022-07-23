from typing import Any, Callable, Mapping, MutableMapping, Optional

import numpy as np
from attrs import define, field

StateConfig = Mapping[str, Any]
StateInputData = Mapping[str, Any]
StateValues = MutableMapping[str, Any]


ProcessingFunction = Callable[["ProcessingState"], Any]


@define
class ProcessingState:
    rng: np.random.Generator
    processing_fun: ProcessingFunction
    config: StateConfig = field(factory=dict)
    input_data: StateInputData = field(factory=dict)
    values: StateValues = field(factory=dict)

    @classmethod
    def create_from_optional(
        cls,
        rng: np.random.Generator,
        processing_fun: ProcessingFunction,
        config: Optional[StateConfig] = None,
        input_data: Optional[StateInputData] = None,
        values: Optional[StateValues] = None,
    ):
        optional_kwargs = {}
        if config is not None:
            # pylint: disable-next=no-member
            optional_kwargs[cls.config.__name__] = config  # type: ignore[attr-defined]
        if input_data is not None:
            optional_kwargs[
                # pylint: disable-next=no-member
                cls.input_data.__name__  # type: ignore[attr-defined]
            ] = input_data
        if values is not None:
            # pylint: disable-next=no-member
            optional_kwargs[cls.values.__name__] = values  # type: ignore[attr-defined]
        return cls(
            rng=rng,
            processing_fun=processing_fun,
            **optional_kwargs,  # type: ignore
        )
