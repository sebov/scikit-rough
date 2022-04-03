from typing import Any, Mapping, MutableMapping

import numpy as np
from attrs import define


@define
class State:
    config: Mapping[str, Any]
    values: MutableMapping[str, Any]
    rng: np.random.Generator
