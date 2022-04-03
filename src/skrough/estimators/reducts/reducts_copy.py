# import typing

# import skrough as rgh


# def get_candidate_attrs_daar(
#     xx,
#     xx_count_distinct,
#     yy,
#     yy_count_distinct,
#     group_index,
#     n_groups,
#     attrs_to_choose_from: np.ndarray,
#     chaos_fun: typing.Callable,
#     settings: typing.Dict,
# ) -> typing.List[int]:
#     if "n_candidate_attrs" in settings:
#         attrs_to_choose_from = np.random.choice(
#             attrs_to_choose_from,
#             np.min([attrs_to_choose_from.size, settings["n_candidate_attrs"]]),
#             replace=False,
#         )

#     scores = np.fromiter(
#         (
#             split_groups_and_compute_chaos_score(
#                 group_index,
#                 n_groups,
#                 i,
#                 xx,
#                 xx_count_distinct,
#                 yy,
#                 yy_count_distinct,
#                 chaos_fun,
#             )
#             for i in attrs_to_choose_from
#         ),
#         dtype=float,
#     )  # type: ignore
#     return attrs_to_choose_from[[scores.argmin()]]  # type: ignore


# def add_attrs_test_daar(
#     xx,
#     xx_count_distinct,
#     yy,
#     yy_count_distinct,
#     group_index,
#     n_groups,
#     attrs,
#     chaos_fun,
#     settings: typing.Dict,
# ) -> typing.List[bool]:
#     result = []
#     for attr in attrs:
#         best_candidate_attr_values = xx[:, attr]
#         best_candidate_attr_chaos_score = split_groups_and_compute_chaos_score_2(
#             group_index,
#             n_groups,
#             best_candidate_attr_values,
#             xx_count_distinct[attr],
#             yy,
#             yy_count_distinct,
#             chaos_fun,
#         )
#         attr_is_better_count = 0
#         for i in range(settings["n_probes_randomness"]):
#             best_candidate_attr_shuffled_values = np.random.permutation(
#                 best_candidate_attr_values
#             )
#             best_candidate_attr_shuffled_chaos_score = (
#                 split_groups_and_compute_chaos_score_2(
#                     group_index,
#                     n_groups,
#                     best_candidate_attr_shuffled_values,
#                     xx_count_distinct[attr],
#                     yy,
#                     yy_count_distinct,
#                     chaos_fun,
#                 )
#             )
#             attr_is_better_count += int(
#                 best_candidate_attr_chaos_score
#                 < best_candidate_attr_shuffled_chaos_score
#             )
#         best_candidate_attr_probe_score = (attr_is_better_count + 1) / (
#             settings["n_probes_randomness"] + 2
#         )
#         result.append(
#             best_candidate_attr_probe_score >= (1 - settings["allowed_randomness"])
#         )
#     return result


# def reduction_phase(
#     xx,
#     xx_count_distinct,
#     yy,
#     yy_count_distinct,
#     group_index,
#     n_groups,
#     chaos_fun,
#     attrs: list[int],
# ):
#     before_reduction_chaos_score = _compute_chaos_score(
#         group_index, n_groups, len(xx), yy, yy_count_distinct, chaos_fun
#     )
#     result_attrs_reduction = attrs[:]
#     for i in reversed(attrs):
#         attrs_to_try = list(result_attrs_reduction)
#         attrs_to_try.remove(i)

#         group_index_reduction = pandas.core.sorting.get_group_index(
#             xx[:, attrs_to_try].T,
#             xx_count_distinct[attrs_to_try],
#             sort=False,
#             xnull=False,
#         )
#         group_index_reduction, _ = pandas.core.sorting.compress_group_index(
#             group_index_reduction, sort=False
#         )
#         n_groups_reduction = max(group_index_reduction) + 1

#         current_chaos_score = _compute_chaos_score(
#             group_index_reduction,
#             n_groups_reduction,
#             len(xx),
#             yy,
#             yy_count_distinct,
#             chaos_fun,
#         )
#         if current_chaos_score <= before_reduction_chaos_score:
#             print("aaaa")
#             result_attrs_reduction = attrs_to_try
#     return result_attrs_reduction


# def grow():
#     pass


# def shrink():
#     pass


# def compute_daar_reduct(
#     xx,
#     xx_count_distinct,
#     yy,
#     yy_count_distinct,
#     chaos_fun: typing.Callable,
#     allowed_randomness,
#     n_probes_randomness,
#     n_candidate_attrs=None,
#     get_candidate_attrs_fun=get_candidate_attrs_daar,
#     add_attrs_test_fun=add_attrs_test_daar,
# ):
#     group_index = np.zeros(xx.shape[0], dtype=np.int_)
#     n_groups = 1
#     all_attrs = np.arange(xx.shape[1])
#     result_attrs = []
#     while True:
#         remaining_attrs = np.delete(all_attrs, result_attrs)
#         if remaining_attrs.size == 0:
#             break

#         candidate_attrs = get_candidate_attrs_fun(
#             xx,
#             xx_count_distinct,
#             yy,
#             yy_count_distinct,
#             group_index,
#             n_groups,
#             remaining_attrs,
#             chaos_fun,
#             {"n_candidate_attrs": n_candidate_attrs},
#         )
#         ##################################################
#         # test if the loop should stop - using attr probes
#         ##################################################
#         attrs_to_add = add_attrs_test_fun(
#             xx,
#             xx_count_distinct,
#             yy,
#             yy_count_distinct,
#             group_index,
#             n_groups,
#             candidate_attrs,
#             chaos_fun,
#             {
#                 "n_probes_randomness": n_probes_randomness,
#                 "allowed_randomness": allowed_randomness,
#             },
#         )
#         if not attrs_to_add[0]:
#             break

#         # TODO: change to use candidate_attrs and attrs_to_add
#         result_attrs.append(int(candidate_attrs[0]))
#         group_index, n_groups = split_groups(
#             group_index,
#             n_groups,
#             xx[:, candidate_attrs[0]],
#             xx_count_distinct[candidate_attrs[0]],
#             compress_group_index=True,
#         )

#     # reduction phase
#     result_attrs = result_attrs + result_attrs[:2]
#     result_attrs_reduction = reduction_phase(
#         xx,
#         xx_count_distinct,
#         yy,
#         yy_count_distinct,
#         group_index,
#         n_groups,
#         chaos_fun,
#         result_attrs,
#     )
#     result_attrs = result_attrs_reduction
#     return result_attrs
