"""Helper functions related to data objects (instances)."""

import logging
from typing import List, Optional, Union

import numpy as np

import skrough.typing as rght
from skrough.logs import log_start_end
from skrough.permutations import get_permutation
from skrough.structs.group_index import GroupIndex
from skrough.unique import get_uniques_positions
from skrough.utils import get_positions_where_values_in
from skrough.weights import prepare_weights

logger = logging.getLogger(__name__)


@log_start_end(logger)
def choose_objects(
    group_index: GroupIndex,
    y: np.ndarray,
    y_count: int,
    objs: Optional[Union[int, float, np.ndarray]] = None,
    weights: Optional[np.ndarray] = None,
    return_representatives_only: bool = False,
    seed: rght.Seed = None,
) -> List[int]:
    """Choose objects having uniform decision values within their groups.

    The function returns a list of objects that have unique decision values within their
    groups in ``group_index`` (see :class:`~skrough.structs.group_index.GroupIndex`).
    There are two ways to specify a hint on how the objects should be selected. If the
    ``objs`` argument is given then its value specifies the subset of objects along with
    their order in which the objects are handled. In such a case only the specified
    objects can appear in the result. Moreover, the objects that are earlier in ``objs``
    take precedence, i.e., if two objects that belongs to the same group (determined by
    ``group_index``) has conflicting decision values then only the earlier one is
    selected for the result. If :code:`objs is None` then all objects represented in
    ``group_index`` are taken into consideration and their order of precedence is
    established by means of the ``weights`` input argument, and the
    :func:`~skrough.weights.prepare_weights` and
    :func:`~skrough.permutations.get_permutation` functions. The value of ``weights`` is
    taken into account only if :code:`objs is None`, otherwise it is ignored. The value
    used during the process of establishing the order of objects is equivalent to the
    following ``selector`` expression::

        proba = prepare_weights(weights, group_index.n_objs, expand_none=False)
        selector = get_permutation(0, group_index.n_objs, proba, seed=seed)

    Thus, giving the opportunity to either set the ``weights`` used for drawing a
    permutation explicitly for each object (when :code:`len(weights) ==
    group_index.n_objs`) or let permutation to be drawn from uniform distribution - see
    :func:`~skrough.weights.prepare_weights` and
    :func:`~skrough.permutations.get_permutation` for details.

    The ``return_representatives_only`` argument is used to control whether the result
    returned by the function should either include all non-conflicting objects (the
    default behavior) or to include at most one object from each group induced by
    ``group_index`` (when :code:`return_representatives_only is True`).

    Args:
        group_index: Group index that represents split of the objects represented by
            this structure into groups.
        y: Factorized decision values for the objects represented by the input
            ``group_index``. The values should be given in a form of integer-location
            based indexing sequence of the factorized decision values, i.e., 0-based
            values that index distinct decisions.
        y_count: Number of distinct decision attribute values.
        objs: A sequence of objects that the function should select from. It should be
            given in a form of integer-location based indexing sequence of the objects
            represented in ``group_index``. :obj:`None` value means to use all available
            objects. Defaults to :obj:`None`.
        weights: Used only if :code:`objs is None`. The value is used for establishing
            the order of precedence of objects by means of the
            :func:`~skrough.weights.prepare_weights` and
            :func:`~skrough.permutations.get_permutation` functions. It should be either
            :code:`len(weights) == group_index.n_objs`, a single weight value or
            :obj:`None`.
        return_representatives_only: A flag controlling if the result should include
            to all non-conflicting objects (when set to ``True``) or to include at most
            one object from each group (when set to :obj:`False`). Defaults to
            :obj:`False`.
        seed: Random seed. Defaults to :obj:`None`.

    Returns:
        A set of objects having uniform decision values within their groups determined
        by ``group_index``. The return value has a form of integer-location based
        indexing sequence of objects represented by ``group_index``.

    Examples:
        >>> group_index = GroupIndex.from_index([0, 0, 1, 1])
        >>> dec = np.array([0, 1, 0, 0])

        >>> choose_objects(group_index, y=dec, y_count=2, objs=np.array([0, 1, 2, 3]))
        [0, 2, 3]

        >>> choose_objects(group_index, y=dec, y_count=2, objs=np.array([1, 0, 2, 3]))
        [1, 2, 3]

        >>> choose_objects(group_index, y=dec, y_count=2, objs=np.array([0, 1, 2, 3]),
        ...                return_representatives_only=True)
        [0, 2]

        >>> choose_objects(group_index, y=dec, y_count=2, objs=np.array([0, 1, 3, 2]),
        ...                return_representatives_only=True)
        [0, 3]

    """
    # TODO: add arguments validation
    # 1) objs is not None => weights is None
    # 2) objs is None and weights is not None => len(group_index.index) == len(weights)

    if len(group_index.index) == 0:
        return []

    if objs is None:
        proba = prepare_weights(weights, group_index.n_objs, expand_none=False)
        selector = get_permutation(0, group_index.n_objs, proba, seed=seed)
    else:
        selector = np.asarray(objs)

    idx = get_uniques_positions(group_index.index[selector])

    representatives_ids = selector[idx]

    if return_representatives_only:
        result = sorted(representatives_ids)
    else:
        group_index_dec = group_index.split(
            y,
            y_count,
            compress=False,
        )
        group_ids = group_index_dec.index[representatives_ids]
        result = get_positions_where_values_in(
            values=group_index_dec.index, reference=group_ids
        )

    return result
