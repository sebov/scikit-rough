import numpy as np

from skrough.chaos_score import get_chaos_score, get_chaos_score_for_group_index
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
):
    chaos_fun = state.config["chaos_fun"]
    base_chaos_score = state.values["base_chaos_score"]
    approx_threshold = state.values["approx_threshold"]

    current_chaos_score = get_chaos_score_for_group_index(
        state.group_index, len(x), y, y_count, chaos_fun
    )
    current_dependency_in_data = base_chaos_score - current_chaos_score
    return current_dependency_in_data >= approx_threshold
