from typing import List, Optional

from attrs import define


@define
class ChaosScoreStats:
    base: float
    total: float
    for_increment_attrs: Optional[List[float]] = None
    approx_threshold: Optional[float] = None
