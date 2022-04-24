# import logging
# from typing import Optional, Sequence, Union, cast

# import numpy as np

# import skrough.typing as rght
# from skrough.algorithms.exceptions import LoopBreak
# from skrough.algorithms.hooks.names import DATA_X, RESULT_ATTRS, SINGLE_GROUP_INDEX
# from skrough.algorithms.meta.helpers import (
#     aggregate_grow_stop_hooks,
#     aggregate_shrink_accept_hooks,
#     normalize_hook_sequence,
#     run_candidates_hooks,
#     aggregate_update_state_hooks,
# )
# from skrough.logs import log_start_end
# from skrough.structs.group_index import GroupIndex
# from skrough.structs.state import GrowShrinkState, StateConfig, StateInput

# logger = logging.getLogger(__name__)


# @log_start_end(logger)
# def grow_shrink(
#     input: StateInput,
#     config: StateConfig,
#     init_hooks: Optional[
#         Union[
#             rght.UpdateStateHook,
#             Sequence[rght.UpdateStateHook],
#         ]
#     ],
#     grow_stop_hooks: Union[
#         rght.StopHook,
#         Sequence[rght.StopHook],
#     ],
#     grow_pre_candidates_hook: rght.PreCandidatesHook,
#     grow_candidates_hooks: Optional[
#         Union[
#             rght.CandidatesHook,
#             Sequence[rght.CandidatesHook],
#         ]
#     ],
#     grow_select_hook: rght.SelectHook,
#     grow_post_select_hooks: Optional[
#         Union[
#             rght.VerifyHook,
#             Sequence[rght.VerifyHook],
#         ]
#     ],
#     shrink_candidate_attrs_hook: Optional[rght.GSShrinkCandidatesHook],
#     shrink_accept_group_index_hooks: Optional[
#         Union[
#             rght.GSShrinkAcceptGroupIndexHook,
#             Sequence[rght.GSShrinkAcceptGroupIndexHook],
#         ]
#     ],
#     finalize_hooks: Optional[
#         Union[
#             rght.UpdateStateHook,
#             Sequence[rght.UpdateStateHook],
#         ]
#     ],
#     prepare_result_hook: rght.PrepareResultHook,
#     seed: rght.Seed = None,
# ):
#     logger.debug("Create state object")
#     rng = np.random.default_rng(seed)
#     state = GrowShrinkState(
#         rng=rng,
#         config=config,
#         input=input,
#     )

#     logger.debug("Normalize init_hooks")
#     init_hooks = normalize_hook_sequence(
#         init_hooks,
#         optional=True,
#     )

#     logger.debug("Normalize grow_stop_hooks")
#     grow_stop_hooks = normalize_hook_sequence(
#         grow_stop_hooks,
#         optional=False,
#     )

#     logger.debug("Normalize get_candidate_attrs_hooks")
#     grow_candidates_hooks = normalize_hook_sequence(
#         grow_candidates_hooks,
#         optional=True,
#     )

#     logger.debug("Normalize post_select_attrs_hooks")
#     grow_post_select_hooks = normalize_hook_sequence(
#         grow_post_select_hooks,
#         optional=True,
#     )

#     logger.debug("Normalize post_select_attrs_hooks")
#     shrink_accept_group_index_hooks = normalize_hook_sequence(
#         shrink_accept_group_index_hooks,
#         optional=True,
#     )

#     logger.debug("Normalize finalize_hooks")
#     finalize_hooks = normalize_hook_sequence(
#         finalize_hooks,
#         optional=True,
#     )

#     grow_stop_check = aggregate_grow_stop_hooks(grow_stop_hooks)

#     shrink_accept_group_index_check = aggregate_shrink_accept_hooks(
#         shrink_accept_group_index_hooks
#     )

#     logger.debug("Run init hooks")
#     aggregate_update_state_hooks(state, init_hooks)

#     ################
#     # grow phase
#     ################
#     logger.info("Start grow phase")
#     try:

#         logger.debug("Check stop conditions")
#         grow_stop_check(state)

#         while True:

#             grow_pre_candidates = grow_pre_candidates_hook(state)

#             grow_candidates = run_candidates_hooks(
#                 state,
#                 grow_pre_candidates,
#                 grow_candidates_hooks,
#             )

#             # select attrs hook
#             logger.debug("Select attrs using select attrs hooks")
#             selected_attrs = grow_select_hook(
#                 state,
#                 grow_candidates,
#             )
#             logger.info("Selected attrs = %s", selected_attrs)

#             logger.debug("Run post select hooks")
#             if grow_post_select_hooks is not None:
#                 for grow_post_select_attrs_hook in grow_post_select_hooks:
#                     selected_attrs = grow_post_select_attrs_hook(
#                         state,
#                         selected_attrs,
#                     )
#             logger.info("Selected attrs after post hooks = %s", selected_attrs)

#             logger.debug("Process selected attrs")
#             if len(selected_attrs) == 0:
#                 logger.debug("Empty selected attrs collection-check stop conditions")
#                 grow_stop_check(state)
#             else:
#                 logger.debug("Add selected attrs one by one")
#                 for selected_attr in selected_attrs:
#                     logger.info("Add attr <%d>", selected_attr)
#                     state.values[RESULT_ATTRS].append(selected_attr)
#                     state.values[SINGLE_GROUP_INDEX] = state.values[
#                         SINGLE_GROUP_INDEX
#                     ].split(
#                         state.values[DATA_X][:, selected_attr],
#                         # TODO: remove this dependency on x_counts
#                         state.values["x_counts"][selected_attr],
#                     )
#                     logger.debug("Check stop conditions")
#                     grow_stop_check(state)

#     except LoopBreak:
#         pass
#     logger.info("End grow phase")
#     ################
#     # end grow phase
#     ################

#     logger.info("Attrs after grow phase = %s", state.values[RESULT_ATTRS])

#     ##################
#     # shrink phase
#     ##################
#     logger.info("Start shrink phase")

#     if shrink_candidate_attrs_hook is None:
#         # TODO: remove cast and code from here
#         shrink_candidate_attrs = cast(
#             rght.Elements, np.asarray(list(reversed(state.values[RESULT_ATTRS])))
#         )
#     else:
#         shrink_candidate_attrs = shrink_candidate_attrs_hook(state)

#     logger.debug("Shrink candidate attrs count = %d", len(shrink_candidate_attrs))

#     for shrink_candidate_attr in shrink_candidate_attrs:
#         shrinked_attrs = state.values[RESULT_ATTRS][:]
#         shrinked_attrs.remove(shrink_candidate_attr)
#         shrinked_group_index = GroupIndex.create_from_data(
#             state.values[DATA_X],
#             # TODO: remove this dependency on x_counts
#             state.values["x_counts"],
#             shrinked_attrs,
#         )
#         if shrink_accept_group_index_check(
#             state,
#             shrinked_group_index,
#         ):
#             logger.info("Removing attr <%d>", shrink_candidate_attr)
#             state.values[RESULT_ATTRS] = shrinked_attrs
#             state.values[SINGLE_GROUP_INDEX] = shrinked_group_index

#     logger.info("End shrink phase")
#     ##################
#     # end shrink phase
#     ##################

#     logger.debug("Run finalize hooks")
#     aggregate_update_state_hooks(state, finalize_hooks)

#     result = prepare_result_hook(state)
#     return result
