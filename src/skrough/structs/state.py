from typing import Any, Callable, Mapping, MutableMapping, Optional

import numpy as np
from attrs import define, field

StateConfig = Mapping[str, Any]
StateInput = Mapping[str, Any]
StateValues = MutableMapping[str, Any]


@define
class ProcessingState:
    rng: np.random.Generator
    processing_fun: Callable[["ProcessingState"], Any]
    config: Optional[StateConfig] = None
    input: Optional[StateInput] = None
    values: StateValues = field(factory=dict)
