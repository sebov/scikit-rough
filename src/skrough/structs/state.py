from typing import Any, List, Mapping, MutableMapping

import numpy as np
from attrs import define, field

from skrough.structs.group_index import GroupIndex

StateConfig = Mapping[str, Any]
StateValues = MutableMapping[str, Any]


@define
class GrowShrinkState:
    group_index: GroupIndex
    rng: np.random.Generator
    result_attrs: List[int] = field(factory=list)
    result_objs: List[int] = field(factory=list)
    config: StateConfig = field(factory=dict)
    values: StateValues = field(factory=dict)
