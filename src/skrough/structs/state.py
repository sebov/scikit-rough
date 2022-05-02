from typing import Any, Mapping, MutableMapping

import numpy as np
from attrs import define, field

StateConfig = Mapping[str, Any]
StateInput = Mapping[str, Any]
StateValues = MutableMapping[str, Any]


@define
class ProcessingState:
    rng: np.random.Generator
    config: StateConfig = field(factory=dict)
    input: StateInput = field(factory=dict)
    values: StateValues = field(factory=dict)
