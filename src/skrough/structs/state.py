from typing import Any, Mapping, MutableMapping, Optional

import numpy as np
from attrs import define, field

StateConfig = Mapping[str, Any]
StateInput = Mapping[str, Any]
StateValues = MutableMapping[str, Any]


@define
class ProcessingState:
    rng: np.random.Generator
    config: Optional[StateConfig] = None
    input: Optional[StateInput] = None
    values: StateValues = field(factory=dict)
