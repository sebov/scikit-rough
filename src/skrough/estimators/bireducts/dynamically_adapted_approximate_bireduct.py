# ''' Bireducts '''

# import sklearn
# import numpy as np

# # from skrough.utils.group_index import split_groups, draw_objects
# # from skrough.metrics.gini_impurity import gini_impurity
# # from skrough.utils.mixin import DecTableOpsMixin
# # from skrough.struct import Bireduct
# # import pandas.core.sorting
# # import pandas.core


# class DynamicallyAdaptedApproximateBireduct(RoughBase, DecTableOpsMixin):


#     def get_bireduct(self):
#         # TODO: check this - is it needed?
#         sklearn.utils.validation.check_is_fitted(self, ['x', 'y'])


#         result_attrs = []
#         while True:


#             best_attr = self.get_best_attr(group_index, n_groups, candidate_attrs,
# xx, xx_count_distinct, yy,
#                                            yy_count_distinct)


#             ###############################################
#             # test the loop should stop - using attr probes
#             ###############################################
#             best_attr_values = xx[:, best_attr]
#             best_attr_chaos_score = self.split_groups_and_compute_chaos_score_2(
# group_index, n_groups,
#   best_attr_values, xx_count_distinct[best_attr],
#      yy, yy_count_distinct)
#             attr_is_better_count = 0
#             for i in range(self.n_of_probes):
#                 best_attr_shuffled_values = np.random.permutation(best_attr_values)
#                 best_attr_shuffled_chaos_score =
# self.split_groups_and_compute_chaos_score_2(group_index, n_groups,
# best_attr_shuffled_values, xx_count_distinct[best_attr],
# yy, yy_count_distinct)
#                 attr_is_better_count +=
# int(best_attr_chaos_score < best_attr_shuffled_chaos_score)

#             best_attr_probe_score = (attr_is_better_count + 1) /
# (self.n_of_probes + 2)

#             if best_attr_probe_score < (1 - self.allowed_randomness):
#                 if len(result_attrs) == 0:
#                     result_attrs.append(int(best_attr))
#                 break
#             ###############################################
#             ###############################################


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
