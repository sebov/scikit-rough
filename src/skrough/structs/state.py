from typing import Any, Mapping, MutableMapping

import numpy as np
from attrs import define, field

StateConfig = Mapping[str, Any]
StateValues = MutableMapping[str, Any]


@define
class GrowShrinkState:
    rng: np.random.Generator
    config: StateConfig = field(factory=dict)
    values: StateValues = field(factory=dict)
