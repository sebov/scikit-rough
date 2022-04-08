# from typing import List

# import skrough.typing as rght
# from skrough.chaos_score import get_chaos_score

# import numpy as np

# from skrough.chaos_score import get_chaos_score_for_group_index
# from skrough.structs.group_index import GroupIndex
# from skrough.structs.attrs_subset import Reduct

# TODO: handle data consistency === chaos
# if check_data_consistency:
#     # by default check_data_consistency should be True
#     # but for now we do not handle this
#     raise NotImplementedError("check_data_consistency==True not implemented")
# else:
#     pass

# TODO: should check sklearn.utils.validation.check_is_fitted(self, ["x", "y"])

# TODO: handle homogeneous groups
# def remove_homogenous_groups(self, group_index, n_groups, y, y_count_distinct):
#     distribution = compute_dec_distribution(group_index, n_groups,
#                                             y, y_count_distinct)
#     groups_homogeneity = compute_homogeneity(distribution)
