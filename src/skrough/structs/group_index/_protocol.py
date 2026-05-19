"""Protocol for group index implementations."""

from typing import Protocol, Self, Sequence, runtime_checkable

import numpy as np
import numpy.typing as npt

import skrough.typing as rght


@runtime_checkable
class GroupIndexProtocol(Protocol):
    """Protocol defining the interface for a group index.

    A group index assigns objects (by their 0-based positions) to groups,
    representing equivalence classes of an indiscernibility relation. Concrete
    implementations must provide methods for splitting groups, computing
    distributions of decision values within groups, and calculating disorder
    scores.
    """

    index: npt.NDArray[np.int64]
    n_groups: int

    @classmethod
    def create_empty(cls) -> Self:
        """Create an empty group index with zero objects."""
        ...

    @classmethod
    def create_uniform(cls, size: int) -> Self:
        """Create a uniform group index where all objects belong to one group."""
        ...

    @classmethod
    def from_index(
        cls,
        index: Sequence[int] | npt.NDArray[np.int64],
        compress: bool = False,
    ) -> Self:
        """Create a group index from a pre-computed group assignment."""
        ...

    @classmethod
    def from_data(
        cls,
        x: npt.NDArray[np.int64],
        x_counts: npt.NDArray[np.int64],
        attrs: rght.IndexListLike | None = None,
    ) -> Self:
        """Split objects into groups according to values on given attributes."""
        ...

    @property
    def n_objs(self) -> int:
        """Number of objects described by this group index."""
        ...

    def split(
        self,
        values: npt.NDArray[np.int64],
        values_count: int,
        compress: bool = False,
    ) -> Self:
        """Split groups of objects into finer groups according to values."""
        ...

    def compress(self) -> Self:
        """Compress group index, removing empty groups."""
        ...

    def get_distribution(
        self,
        values: npt.NDArray[np.int64],
        values_count: int,
    ) -> npt.NDArray[np.int64]:
        """Compute decision distribution within groups of objects."""
        ...

    def get_disorder_score(
        self,
        values: npt.NDArray[np.int64],
        values_count: int,
        disorder_fun: rght.DisorderMeasure,
    ) -> rght.DisorderMeasureReturnType:
        """Compute disorder score for the given grouping of objects."""
        ...

    def get_disorder_score_after_split(
        self,
        split_values: npt.NDArray[np.int64],
        split_values_count: int,
        values: npt.NDArray[np.int64],
        values_count: int,
        disorder_fun: rght.DisorderMeasure,
    ) -> rght.DisorderMeasureReturnType:
        """Compute disorder score after splitting by additional attribute."""
        ...
