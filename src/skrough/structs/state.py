from typing import Any, Mapping, MutableMapping

import numpy as np
from attrs import define

StateConfig = Mapping[str, Any]
StateValues = MutableMapping[str, Any]


@define
class State:
    config: StateConfig
    values: StateValues
    rng: np.random.Generator
