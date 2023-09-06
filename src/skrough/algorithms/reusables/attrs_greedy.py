# pylint: disable=duplicate-code

from __future__ import annotations

from skrough.algorithms import hooks
from skrough.algorithms.key_names import CONFIG_CANDIDATES_SELECT_RANDOM_MAX_COUNT
from skrough.algorithms.meta import stage

attrs_greedy_stage = stage.Stage.from_hooks(
    stop_hooks=[
        hooks.stop_hooks.stop_hook_approx_threshold,
    ],
    init_hooks=None,
    pre_candidates_hooks=[
        hooks.pre_candidates_hooks.pre_candidates_hook_remaining_attrs,
    ],
    candidates_hooks=[
        hooks.common.process_elements.create_process_elements_hook_random_choice(
            CONFIG_CANDIDATES_SELECT_RANDOM_MAX_COUNT
        )
    ],
    select_hooks=[
        hooks.select_hooks.select_hook_attrs_chaos_score_based,
    ],
    filter_hooks=None,
    inner_init_hooks=None,
    inner_stop_hooks=hooks.inner_stop_hooks.inner_stop_hook_empty,
    inner_process_hooks=hooks.inner_process_hooks.inner_process_hook_add_first_attr,
    finalize_hooks=None,
)
