# pylint: disable=unused-argument

import numpy as np

from skrough.chaos_score import get_chaos_score, get_chaos_score_for_group_index
from skrough.instances import choose_objects
from skrough.structs.bireduct import Bireduct
from skrough.structs.reduct import Reduct
from skrough.structs.state import GrowShrinkState


def init_state_approx_threshold(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
):
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


def check_stop_approx_threshold(
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


def check_stop_len(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
) -> bool:
    return len(state.result_attrs) >= state.config["result_attrs_max_len"]


def check_stop_empty_add_attrs(
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


def get_candidate_attrs_random(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
    input_attrs: np.ndarray,
) -> np.ndarray:
    candidate_attrs_len = state.config["candidate_attrs_max_len"]
    candidate_attrs = state.rng.choice(
        input_attrs,
        min(len(input_attrs), candidate_attrs_len),
        replace=False,
    )
    return candidate_attrs


def select_attrs_random(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
    input_attrs: np.ndarray,
) -> np.ndarray:
    return state.rng.choice(input_attrs, state.config["select_attrs_random_count"])


def select_attrs_gain_based(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
    input_attrs: np.ndarray,
):
    pass


def post_select_attrs_empty(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
    input_attrs: np.ndarray,
):
    if len(input_attrs) == 0:
        value = state.values.get("empty_add_attrs_count", 0) + 1
    else:
        value = 0
    state.values["empty_add_attrs_count"] = value
    print(state.values)


def prepare_result_reduct(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
) -> Reduct:
    return Reduct(attrs=state.result_attrs)


def prepare_result_bireduct(
    x: np.ndarray,
    x_counts: np.ndarray,
    y: np.ndarray,
    y_count: int,
    state: GrowShrinkState,
) -> Bireduct:
    result_objs = choose_objects(
        state.group_index,
        y,
        y_count,
        seed=state.rng,
    )
    return Bireduct(objs=result_objs, attrs=state.result_attrs)
