# pylint: disable=duplicate-code

from __future__ import annotations

from skrough.algorithms import hooks
from skrough.algorithms.meta import stage

objs_choose_randomly = stage.Stage.from_hooks(
    stop_hooks=hooks.stop_hooks.stop_hook_always_true,
    init_hooks=None,
    pre_candidates_hooks=None,
    candidates_hooks=None,
    select_hooks=None,
    filter_hooks=None,
    inner_init_hooks=None,
    inner_stop_hooks=hooks.inner_stop_hooks.inner_stop_hook_empty,
    inner_process_hooks=hooks.inner_process_hooks.inner_process_hook_add_first_attr,
    finalize_hooks=hooks.finalize_hooks.finalize_hook_choose_objs_randomly,
)
