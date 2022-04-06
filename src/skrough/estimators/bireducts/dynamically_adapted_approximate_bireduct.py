# ''' Bireducts '''


#         # reduction phase
#         before_reduction_chaos_score = self.compute_chaos_score(group_index, n_groups,
# xx, yy,
#                                                            yy_count_distinct)
#         result_attrs_reduction = list(result_attrs)
#         # print(len(result_attrs))
#         if len(result_attrs) > 1:
#             for i in reversed(result_attrs):
#                 attrs_to_try = list(result_attrs_reduction)
#                 attrs_to_try.remove(i)

#                 group_index_reduction = pandas.core.sorting.get_group_index(
# xx[:, attrs_to_try].T,
# xx_count_distinct[attrs_to_try], sort=False,
# xnull=False)
#                 group_index_reduction, _ = pandas.core.sorting.compress_group_index(
# group_index_reduction, sort=False)
#                 n_groups_reduction = max(group_index_reduction) + 1

#                 current_chaos_score = self.compute_chaos_score(group_index_reduction,
# n_groups_reduction, xx, yy,
#                                                                yy_count_distinct)
#                 if current_chaos_score <= before_reduction_chaos_score:
#                     result_attrs_reduction = attrs_to_try

#         # print(len(result_attrs_reduction))
#         # print('----')

#         # update group_index
#         result_attrs = result_attrs_reduction
#         group_index = pandas.core.sorting.get_group_index(xx[:, result_attrs].T,
# xx_count_distinct[result_attrs], sort=False,
# xnull=False)
#         group_index, _ = pandas.core.sorting.compress_group_index(group_index,
# sort=False)
#         n_groups = max(group_index) + 1
