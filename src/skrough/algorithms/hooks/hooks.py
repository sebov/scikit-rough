# pylint: disable=unused-argument

import logging

import numpy as np

import skrough.typing as rght
from skrough.chaos_score import get_chaos_score, get_chaos_score_for_group_index
from skrough.instances import choose_objects
from skrough.structs.attrs_subset import AttrsSubset
from skrough.structs.group_index import GroupIndex
from skrough.structs.objs_attrs_subset import ObjsAttrsSubset
from skrough.structs.state import GrowShrinkState

DEFAULT_DAAR_SMOOTHING_PARAMETER = 1

logger = logging.getLogger(__name__)


def _split_groups_and_compute_chaos_score(
    group_index: GroupIndex,
    attr_values: np.ndarray,
    attr_count: int,
    y: np.ndarray,
    y_count: int,
    chaos_fun: rght.ChaosMeasure,
):
    tmp_group_index = group_index.split(attr_values, attr_count)
    return get_chaos_score_for_group_index(
        tmp_group_index, len(attr_values), y, y_count, chaos_fun
    )


def init_state_approx_threshold(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
) -> None:
    chaos_fun = state.config["chaos_fun"]
    epsilon = state.config["epsilon"]

    # compute base chaos score
    base_chaos_score = get_chaos_score(
        x,
        x_counts,
        y,
        y_count,
        [],
        chaos_fun=chaos_fun,
    )

    # compute total chaos score
    total_chaos_score = get_chaos_score(
        x,
        x_counts,
        y,
        y_count,
        range(x.shape[1]),
        chaos_fun=chaos_fun,
    )
    total_dependency_in_data = base_chaos_score - total_chaos_score
    approx_threshold = (1 - epsilon) * total_dependency_in_data - np.finfo(float).eps
    state.values.update(
        {
            "base_chaos_score": base_chaos_score,
            "approx_threshold": approx_threshold,
        }
    )


def grow_stop_approx_threshold(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
) -> bool:
    chaos_fun = state.config["chaos_fun"]
    base_chaos_score = state.values["base_chaos_score"]
    approx_threshold = state.values["approx_threshold"]
    current_chaos_score = get_chaos_score_for_group_index(
        state.group_index, len(x), y, y_count, chaos_fun
    )
    current_dependency_in_data = base_chaos_score - current_chaos_score
    return current_dependency_in_data >= approx_threshold


def grow_stop_count(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
) -> bool:
    return len(state.result_attrs) >= state.config["result_attrs_max_count"]


def grow_stop_empty_add_attrs(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
) -> bool:
    return (
        state.values.get("empty_add_attrs_count", 0)
        >= state.config["empty_add_attrs_count_max"]
    )


def grow_candidate_attrs_random(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
    input_attrs: np.ndarray,
) -> np.ndarray:
    candidate_attrs_count = state.config.get("candidate_attrs_max_count")
    if candidate_attrs_count is None:
        candidate_attrs = input_attrs.copy()
    else:
        candidate_attrs = state.rng.choice(
            input_attrs,
            min(len(input_attrs), candidate_attrs_count),
            replace=False,
        )
    return candidate_attrs


def grow_select_attrs_random(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
    input_attrs: np.ndarray,
) -> np.ndarray:
    return state.rng.choice(input_attrs, state.config["select_attrs_random_count"])


def grow_select_attrs_gain_based(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
    input_attrs: np.ndarray,
):
    chaos_fun = state.config["chaos_fun"]
    attrs_count = state.config["select_attrs_gain_based_count"]
    scores = np.fromiter(
        (
            _split_groups_and_compute_chaos_score(
                state.group_index, x[:, i], x_counts[i], y, y_count, chaos_fun
            )
            for i in input_attrs
        ),
        dtype=float,
    )
    # find indices for which the scores are the lowest
    selected_attrs_idx = np.argsort(scores)[:attrs_count]
    return input_attrs[selected_attrs_idx]


def _check_if_better_than_shuffled(
    group_index: GroupIndex,
    attr_values: np.ndarray,
    attr_count: int,
    y: np.ndarray,
    y_count: int,
    n_of_probes: int,
    smoothing_parameter: float,
    allowed_randomness: float,
    chaos_fun: rght.ChaosMeasure,
    rng: np.random.Generator,
) -> bool:
    logger.debug("Start %s function", _check_if_better_than_shuffled.__name__)
    attr_chaos_score = _split_groups_and_compute_chaos_score(
        group_index,
        attr_values,
        attr_count,
        y,
        y_count,
        chaos_fun,
    )
    attr_is_better_count = 0
    for _ in range(n_of_probes):
        attr_values_shuffled = rng.permutation(attr_values)
        shuffled_chaos_score = _split_groups_and_compute_chaos_score(
            group_index,
            attr_values_shuffled,
            attr_count,
            y,
            y_count,
            chaos_fun,
        )
        attr_is_better_count += int(attr_chaos_score < shuffled_chaos_score)

    smoothing_dims = 2  # binomial distribution, i.e., better/worse
    attr_probe_score = (attr_is_better_count + smoothing_parameter) / (
        n_of_probes + smoothing_parameter * smoothing_dims
    )
    logger.debug("attr_probe_score == %f", attr_probe_score)
    logger.debug("End %s function", _check_if_better_than_shuffled.__name__)
    return attr_probe_score >= (1 - allowed_randomness)


def grow_post_select_attrs_daar(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
    input_attrs: np.ndarray,
) -> np.ndarray:
    logger.debug("Start %s function", grow_post_select_attrs_daar.__name__)
    daar_n_of_probes = state.config["post_select_attrs_daar_n_of_probes"]
    logger.debug("Param daar_n_of_probes == %d", daar_n_of_probes)
    daar_smoothing_parameter = state.config.get(
        "post_select_attrs_daar_smoothing_parameter",
        DEFAULT_DAAR_SMOOTHING_PARAMETER,
    )
    logger.debug("Param daar_smoothing_parameter == %f", daar_smoothing_parameter)
    daar_allowed_randomness = state.config["post_select_attrs_daar_allowed_randomness"]
    logger.debug("Param daar_allowed_randomness == %f", daar_allowed_randomness)
    chaos_fun = state.config["chaos_fun"]
    result = []
    for input_attr in input_attrs:
        logger.debug("Check if attr <%d> is better than shuffled", input_attr)
        if _check_if_better_than_shuffled(
            state.group_index,
            x[:, input_attr],
            x_counts[input_attr],
            y,
            y_count,
            daar_n_of_probes,
            daar_smoothing_parameter,
            daar_allowed_randomness,
            chaos_fun,
            state.rng,
        ):
            logger.debug(
                "Attr <%d> is better than shuffled with respect to allowed_randomness",
                input_attr,
            )
            result.append(input_attr)
    logger.debug("End %s function", grow_post_select_attrs_daar.__name__)
    return np.asarray(result)


def grow_post_select_attrs_check_empty(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
    input_attrs: np.ndarray,
) -> np.ndarray:
    if len(input_attrs) == 0:
        value = state.values.get("empty_add_attrs_count", 0) + 1
    else:
        value = 0
    state.values["empty_add_attrs_count"] = value
    return input_attrs


def shrink_candidate_attrs_reversed(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
) -> np.ndarray:
    return np.asarray(list(reversed(state.result_attrs)))


def shrink_accept_group_index_approx_threshold(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
    group_index_to_check: GroupIndex,
) -> bool:
    chaos_fun = state.config["chaos_fun"]
    base_chaos_score = state.values["base_chaos_score"]
    approx_threshold = state.values["approx_threshold"]
    current_chaos_score = get_chaos_score_for_group_index(
        group_index_to_check, len(x), y, y_count, chaos_fun
    )
    current_dependency_in_data = base_chaos_score - current_chaos_score
    return current_dependency_in_data >= approx_threshold


def finalize_state_draw_objects(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
) -> None:
    logger.debug("Start %s function", finalize_state_draw_objects.__name__)
    result_objs = choose_objects(
        state.group_index,
        y,
        y_count,
        seed=state.rng,
    )
    logger.debug("Chosen objects count = %d", len(result_objs))
    state.result_objs = result_objs
    logger.debug("End %s function", finalize_state_draw_objects.__name__)


def prepare_result_attrs_subset(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
) -> AttrsSubset:
    return AttrsSubset(attrs=state.result_attrs)


def prepare_result_objs_attrs_subset(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
) -> ObjsAttrsSubset:
    return ObjsAttrsSubset(objs=state.result_objs, attrs=state.result_attrs)
