from typing import Any, Optional, Union

import numpy as np

import skrough.typing as rght
from skrough.chaos_measures.chaos_measures import conflicts_number
from skrough.chaos_score import get_chaos_score_stats
from skrough.dataprep import prepare_factorized_values, prepare_factorized_x
from skrough.instances import choose_objects
from skrough.structs.group_index import GroupIndex
from skrough.typing_utils import unify_attrs, unify_objs
from skrough.unique import get_rows_nunique


def check_if_functional_dependency(
    x: np.ndarray,
    y: np.ndarray,
    objs: Optional[rght.ObjsLike] = None,
    attrs: Optional[rght.AttrsLike] = None,
) -> bool:
    """Check functional dependency between conditional attributes and the decision.

    Check functional dependency between conditional attributes and the decision. The
    check is based on the number of duplicated rows induced by the given subset of
    attributes either with or without the decision attribute. If the number of
    duplicated rows is the same, the functional dependency holds. The check can be
    further narrowed to the given subset of attributes and objects.

    Args:
        x: Input data table.
        y: Input decision.
        objs: A subset of objects that the check should be performed on. It should
            be given in a form of a sequence of integer-location based indexing of the
            selected objects/rows/instances from ``x``. ``None`` value means to use
            all available objects. Defaults to None.
        attrs: A subset of conditional attributes the check should be performed on.
            It should be given in a form of a sequence of integer-location based
            indexing of the selected conditional attributes from ``x``. ``None`` value
            means to use all available conditional attributes. Defaults to None.

    Returns:
        Indication whether functional dependency holds for the given input.
    """
    unified_objs: Union[rght.Objs, slice] = (
        unify_objs(objs) if objs is not None else slice(None)
    )
    unified_attrs: Union[rght.Attrs, slice] = (
        unify_attrs(attrs) if attrs is not None else slice(None)
    )
    x_index_expr: Any
    if isinstance(unified_objs, slice) or isinstance(unified_attrs, slice):
        x_index_expr = np.index_exp[unified_objs, unified_attrs]
    else:
        # we want to take all ``objects`` x ``attributes``
        x_index_expr = np.ix_(unified_objs, unified_attrs)
    data = x[x_index_expr]
    nunique = get_rows_nunique(data)
    data = np.column_stack((data, y[unified_objs]))
    nunique_with_dec = get_rows_nunique(data)
    return nunique == nunique_with_dec


def check_if_consistent_table(
    x: np.ndarray,
    y: np.ndarray,
) -> bool:
    """Check if decision table is consistent.

    Check if decision table is consistent, i.e., check if it is possible to discern
    objects with different decisions by means of conditional attributes. It is realized
    just as a simple wrapper around ``check_if_functional_dependency`` function
    using all available objects and attributes.

    Args:
        x: Input data table.
        y: Input decision.

    Returns:
        Indication whether the decision table is consistent.
    """
    return check_if_functional_dependency(x, y)


def check_if_reduct(
    x: np.ndarray,
    y: np.ndarray,
    attrs: rght.AttrsLike,
    consistent_table_check: bool = True,
) -> bool:
    """Check if specified attributes form a reduct.

    _extended_summary_

    Args:
        x: Input data table.
        y: Input decision.
        attrs: A subset of conditional attributes the check should be performed on.
            It should be given in a form of a sequence of integer-location based
            indexing of the selected conditional attributes from ``x``.
        consistent_table_check: Whether decision table consistency check should be
            performed prior to other checks. Defaults to True.

    Returns:
        Indication whether the specified attributes form a reduct.
    """
    if len(set(attrs)) < len(attrs):
        raise ValueError("duplicated attrs in the given sequence")

    if consistent_table_check and not check_if_consistent_table(x, y):
        return False

    x, x_counts = prepare_factorized_x(x)
    y, y_count = prepare_factorized_values(y)
    group_index = GroupIndex.create_from_data(x, x_counts)
    base_conflicts = group_index.get_chaos_score(y, y_count, conflicts_number)

    candidate_group_index = GroupIndex.create_from_data(x, x_counts, attrs)
    candidate_conflicts = candidate_group_index.get_chaos_score(
        y, y_count, conflicts_number
    )
    if base_conflicts != candidate_conflicts:
        return False

    all_cols = set(attrs)
    for i in attrs:
        reduced_group_index = GroupIndex.create_from_data(
            x, x_counts, list(all_cols - {i})
        )
        reduced_conflicts = reduced_group_index.get_chaos_score(
            y, y_count, conflicts_number
        )
        if reduced_conflicts == base_conflicts:
            return False

    return True


def check_if_bireduct(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    objs: rght.ObjsLike,
    attrs: rght.AttrsLike,
) -> bool:
    """Check if specified objects and attributes form a bireduct.

    _extended_summary_

    Args:
        x: Input data table.
        x_counts: _description_
        y: Input decision.
        y_count: _description_
        objs: _description_
        attrs: _description_

    Returns:
        Indication whether the specified objects and attributes form a reduct.
    """
    if not check_if_functional_dependency(x, y, objs, attrs):
        return False
    group_index = GroupIndex.create_from_data(x, x_counts, attrs)
    all_objs = np.concatenate((objs, np.arange(len(x))))
    chosen_objs = choose_objects(group_index, y, y_count, all_objs)
    return set(chosen_objs) == set(objs)


def check_if_approx_reduct(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    attrs: rght.AttrsLike,
    chaos_fun: rght.ChaosMeasure,
    epsilon: float,
) -> bool:
    chaos_score_stats = get_chaos_score_stats(
        x,
        x_counts,
        y,
        y_count,
        chaos_fun=chaos_fun,
        epsilon=epsilon,
        increment_attrs=[attrs],
    )
    if not chaos_score_stats.for_increment_attrs:
        raise ValueError("Chaos score increment attrs should not be empty")
    if chaos_score_stats.approx_threshold is None:
        raise ValueError("Chaos score approx threshold should not be empty")

    return (
        # pylint: disable-next=unsubscriptable-object
        chaos_score_stats.for_increment_attrs[0]
        <= chaos_score_stats.approx_threshold
    )
