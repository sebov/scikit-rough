''' Bireducts '''

import sklearn
import numpy as np

from skrough.base import RoughBase
from skrough.utils.group_index import split_groups
from skrough.metrics.gini_impurity import gini_impurity
from skrough.utils.mixin import DecTableOpsMixin
from skrough.base import Reduct


class GreedyHeuristicReduct(RoughBase, DecTableOpsMixin):

    def __init__(self, epsilon=0.0, n_candidate_attrs=None, score_func=gini_impurity, random_state=None):
        super(GreedyHeuristicReduct, self).__init__(score_func, random_state=random_state)
        self.epsilon = epsilon
        self.n_candidate_attrs = n_candidate_attrs

    def fit(self, x, y, sample_weight=None, check_data_consistency=True):
        super(GreedyHeuristicReduct, self).fit(
            x, y,
            sample_weight=sample_weight,
            check_data_consistency=check_data_consistency
        )
        if not 0 <= self.epsilon <= 1:
            raise ValueError(f'epsilon must be in (0, 1], got {self.epsilon}')

        return self

    def get_best_attr(self, group_index, n_groups, candidate_attrs, x, x_count_distinct, y, y_count_distinct):
        scores = np.fromiter((self.split_groups_and_compute_chaos_score(group_index, n_groups, i, x, x_count_distinct, y, y_count_distinct)
                              for i in candidate_attrs), dtype=float)
        return candidate_attrs[scores.argmin()]

    def get_reduct(self):
        # TODO: check this - is it needed?
        sklearn.utils.validation.check_is_fitted(self, ['x', 'y'])

        if not self.check_data_consistency:
            total_chaos_score = 0

        group_index = np.zeros(len(self.x), dtype=np.int_)
        n_groups = 1
        decision_chaos_score = self.compute_chaos_score(group_index, n_groups, self.x, self.y, self.y_count_distinct)
        total_dependency_in_data = decision_chaos_score - total_chaos_score
        approx_threshold = (1 - self.epsilon) * total_dependency_in_data - np.finfo(float).eps

        result_attrs = []
        while True:
            print(f'iteration-{len(result_attrs)} {n_groups}')
            current_chaos_score = self.compute_chaos_score(group_index, n_groups, self.x, self.y, self.y_count_distinct)
            current_dependency_in_data = decision_chaos_score - current_chaos_score
            if current_dependency_in_data >= approx_threshold:
                break
            candidate_attrs = np.delete(np.arange(self.x.shape[1]), result_attrs)
            if self.n_candidate_attrs is not None:
                # TODO: introduce random_state usage
                candidate_attrs = np.random.choice(candidate_attrs,
                                                   np.min([len(candidate_attrs), self.n_candidate_attrs]),
                                                   replace=False)
            best_attr = self.get_best_attr(group_index, n_groups, candidate_attrs, self.x, self.x_count_distinct, self.y, self.y_count_distinct)
            result_attrs.append(best_attr)
            group_index, n_groups = split_groups(group_index,
                                                 n_groups,
                                                 self.x[:, best_attr],
                                                 self.x_count_distinct[best_attr],
                                                 compress_group_index=True)

        # group_index = np.zeros(len(self.x), dtype=np.int_)
        # n_groups = 1
        # for attr in result_attrs:
        #     group_index, n_groups = split_groups(group_index,
        #                                               n_groups,
        #                                               self.x[:, attr],
        #                                               self.x_count_distinct[attr],
        #                                               compress_group_index=True)

        return Reduct(result_attrs)

