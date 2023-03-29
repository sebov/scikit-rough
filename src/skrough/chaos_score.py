"""Functions related to chaos/disorder score in data."""

import logging
from typing import Optional, Sequence, Set

import numpy as np

import skrough.typing as rght
from skrough.logs import log_start_end
from skrough.structs.chaos_score_stats import ChaosScoreStats
from skrough.structs.group_index import GroupIndex

logger = logging.getLogger(__name__)


@log_start_end(logger)
def get_chaos_score_for_data(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    chaos_fun: rght.ChaosMeasure,
    attrs: Optional[rght.LocationsLike] = None,
) -> rght.ChaosMeasureReturnType:
    """Compute chaos score induced by the given subset of attributes.

    Compute chaos score value for the grouping (equivalence classes) induced by the
    given subset of attributes.

    Args:
        x: Factorized data table representing conditional features/attributes for the
            objects the computation should be performed on. The values in each column
            should be given in a form of integer-location based indexing sequence of the
            factorized conditional attribute values, i.e., 0-based values that index
            distinct values of the conditional attribute.
        x_counts: Number of distinct attribute values given for each conditional
            attribute. The argument is expected to be given as a 1D array.
        y: Factorized decision values for the objects represented by the input
            :obj:`x` argument. The values should be given in a form of integer-location
            based indexing sequence of the factorized decision values, i.e., 0-based
            values that index distinct decisions.
        y_count: Number of distinct decision attribute values.
        chaos_fun: Chaos measure function to be used for computing the chaos score.
        attrs: A subset of conditional attributes the chaos score should be computed
            for. It should be given in a form of a sequence of integer-location based
            indexing of the selected conditional attributes from ``x``. :obj:`None`
            value means to use all available conditional attributes. Defaults to
            :obj:`None`.

    Returns:
        Chaos score value obtained for the grouping (equivalence classes) induced by the
        given subset of attributes.
    """
    group_index = GroupIndex.from_data(x, x_counts, attrs)
    result = group_index.get_chaos_score(y, y_count, chaos_fun)
    return result


@log_start_end(logger)
def get_chaos_score_stats(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    chaos_fun: rght.ChaosMeasure,
    increment_attrs: Optional[Sequence[rght.LocationsLike]] = None,
    epsilon: Optional[float] = None,
) -> ChaosScoreStats:
    """Compute chaos score stats.

    Compute chaos score stats for the given input data. The function computes the
    following results:

    - base chaos score - the chaos score value for the situation when all objects
      are considered to be just in one group
    - total chaos score - the chaos score value for the grouping (equivalence
      classes) induced by all conditional attributes
    - approximation threshold (optional result) - the chaos score value that should be
      considered as the goal/limit of some heuristic process aiming at minimizing the
      chaos score below the threshold. The approximation threshold is a value somewhere
      between ``total`` (low value) and ``base`` (high value) and it is established by
      means of the ``epsilon`` input argument
    - intermediate chaos score values (optional result) - a non-increasing sequence of
      chaos score values obtained for a growing subset of attributes defined by the
      ``increment_attrs`` input attribute. It should be understood as a sequence of
      chaos score values computed for groupings induced by cumulative sum of attribute
      subsets defined by ``incremental_attrs``. E.g., if :code:`increment_attrs == [[2,
      7], [1], [8]]` then the returned intermediate chaos score result will consist of
      the chaos score values computed for the following attribute subsets :code:`[2, 7],
      [2, 7, 1], [2, 7, 1, 8]`, respectively. The function handles inputs with repeated
      attributes properly, e.g., the elements of ``increment_attrs`` need not to be
      disjoint (for whatever reason) and the results obtained for :code:`[[0], [1]]` vs.
      :code:`[[0], [0, 1]]` will be the same.

    Args:
        x: Factorized data table representing conditional features/attributes for the
            objects the computation should be performed on. The values in each column
            should be given in a form of integer-location based indexing sequence of the
            factorized conditional attribute values, i.e., 0-based values that index
            distinct values of the conditional attribute.
        x_counts: Number of distinct attribute values given for each conditional
            attribute. The argument is expected to be given as a 1D array.
        y: Factorized decision values for the objects represented by the input
            :obj:`x` argument. The values should be given in a form of integer-location
            based indexing sequence of the factorized decision values, i.e., 0-based
            values that index distinct decisions.
        y_count: Number of distinct decision attribute values.
        chaos_fun: Chaos measure function to be used for computing the chaos score.
        increment_attrs: A sequence of attribute subsets that defines a sequence of
            growing subsets of attributes obtained as a cumulative sum (in the set
            theoretic way) of this input. E.g.::

                increment_attrs == [[2, 7], [1], [8]]
                cumulative_sum == [[2, 7], [2, 7, 1], [2, 7, 1, 8]]

            The latter sequence is then used to compute a non-increasing sequence of
            chaos score values using its elements (i.e., subsets of attributes) to split
            objects into equivalence classes with respect to indiscernibility relation.
            When set to :obj:`None` then the incremental attrs processing is not
            performed. Defaults to :obj:`None`.
        epsilon: A value :code:`0.0 <= epsilon <= 1.0` used to compute the resulting
            approximation threshold using the expression equivalent to::

                delta_dependency = base_chaos_score - total_chaos_score
                approx_threshold = total_chaos_score + epsilon * delta_dependency + A

            Where ``A`` is a very small number and it is added to overcome some possible
            floating-point arithmetic issues. When set to :obj:`None` then
            ``approx_threshold`` is not computed. Defaults to :obj:`None`.

    Returns:
        :class:`~skrough.structs.chaos_score_stats.ChaosScoreStats` instance
        representing statistics computed by the function.

    Examples:
        >>> from skrough.chaos_measures import entropy
        >>> from attrs import asdict
        >>> x, x_counts = prepare_factorized_array(np.asarray([[8, 8, 8],
        ...                                                    [8, 8, 8],
        ...                                                    [1, 8, 8],
        ...                                                    [1, 1, 8],
        ...                                                    [1, 1, 1]]))
        >>> y, y_count = prepare_factorized_vector(np.asarray([3, 4, 3, 4, 5]))
        >>> res = get_chaos_score_stats(x, x_counts, y, y_count, chaos_fun=entropy,
        ...                             increment_attrs=[[0], [2]],
        ...                             epsilon=0.2)
        >>> type(res)
        skrough.structs.chaos_score_stats.ChaosScoreStats
        >>> asdict(res)
        {'base': 1.5219280948873621,
        'total': 0.4,
        'for_increment_attrs': [1.3509775004326936, 0.8],
        'approx_threshold': 0.6243856189774726}
    """

    if epsilon is not None and (epsilon < 0 or epsilon > 1):
        raise ValueError(
            "Epsilon value should be a number between 0.0 and 1.0 inclusive"
        )

    group_index = GroupIndex.create_uniform(len(x))

    # compute base chaos score
    base_chaos_score = group_index.get_chaos_score(y, y_count, chaos_fun)

    increment_attrs_chaos_score = None
    attrs_added: Set[int] = set()
    if increment_attrs is not None:
        increment_attrs_chaos_score = []
        for attrs in increment_attrs:
            attrs_to_add = set(attrs) - attrs_added
            for attr in attrs_to_add:
                group_index = group_index.split(
                    x[:, attr], x_counts[attr], compress=True
                )
            attrs_added = attrs_added.union(attrs_to_add)
            chaos_score = group_index.get_chaos_score(y, y_count, chaos_fun)
            increment_attrs_chaos_score.append(chaos_score)

    # add remaining attrs
    attrs_other = set(range(x.shape[1])) - attrs_added
    for attr in attrs_other:
        group_index = group_index.split(x[:, attr], x_counts[attr], compress=True)

    # compute total chaos score
    total_chaos_score = group_index.get_chaos_score(y, y_count, chaos_fun)

    approx_threshold = None
    if epsilon is not None:
        delta_dependency = base_chaos_score - total_chaos_score
        approx_threshold = np.nextafter(
            total_chaos_score + epsilon * delta_dependency,
            np.inf,
        )

    result = ChaosScoreStats(
        base=base_chaos_score,
        total=total_chaos_score,
        for_increment_attrs=increment_attrs_chaos_score,
        approx_threshold=approx_threshold,
    )
    logger.debug("chaos_stats = %s", result)
    return result
