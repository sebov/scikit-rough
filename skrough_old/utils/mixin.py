import numpy as np
import pandas as pd
import sklearn

from skrough.utils.group_index import compute_dec_distribution, compute_homogeneity, split_groups


class DecTableOpsMixin(object):

    def compute_chaos_score(self, group_index, n_groups, x, y, y_count_distinct):
        distribution = compute_dec_distribution(group_index, n_groups, y, y_count_distinct)
        return self.score_func(distribution, len(x))

    def split_groups_and_compute_chaos_score(self, group_index, n_groups, attr, x, x_count_distinct, y, y_count_distinct):
        tmp_group_index, tmp_n_groups = split_groups(group_index,
                                                     n_groups,
                                                     x[:, attr],
                                                     x_count_distinct[attr])
        return self.compute_chaos_score(tmp_group_index, tmp_n_groups, x, y, y_count_distinct)

    def split_groups_and_compute_chaos_score_2(self, group_index, n_groups, attr_values, attr_count_distinct, y, y_count_distinct):
        tmp_group_index, tmp_n_groups = split_groups(group_index,
                                                     n_groups,
                                                     attr_values,
                                                     attr_count_distinct)
        return self.compute_chaos_score(tmp_group_index, tmp_n_groups, attr_values, y, y_count_distinct)

    def remove_homogenous_groups(self, group_index, n_groups, y, y_count_distinct):
        distribution = compute_dec_distribution(group_index, n_groups, y, y_count_distinct)
        groups_homogeneity = compute_homogeneity(distribution)

