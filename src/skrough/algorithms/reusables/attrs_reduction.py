from skrough.algorithms import hooks
from skrough.algorithms.meta import stage

attrs_reduction_stage = stage.Stage.from_hooks(
    stop_hooks=[
        hooks.stop_hooks.stop_hook_always_false,
    ],
    init_hooks=hooks.init_hooks.init_hook_current_approx_threshold,
    pre_candidates_hooks=[hooks.pre_candidates_hooks.pre_candidates_hook_result_attrs],
    candidates_hooks=[
        hooks.common.process_elements.process_elements_hook_reverse_elements
    ],
    select_hooks=[
        hooks.common.process_elements.process_elements_hook_pass_everything,
    ],
    filter_hooks=None,
    inner_init_hooks=None,
    inner_stop_hooks=hooks.inner_stop_hooks.inner_stop_hook_empty_loop_break,
    inner_process_hooks=(
        hooks.inner_process_hooks.inner_process_hook_discard_first_attr_approx_threshold
    ),
    finalize_hooks=None,
)
