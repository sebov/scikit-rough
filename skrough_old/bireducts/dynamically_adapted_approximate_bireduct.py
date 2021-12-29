''' Bireducts '''

import sklearn
import numpy as np

from skrough.base import RoughBase
from skrough.utils.group_index import split_groups, draw_objects
from skrough.metrics.gini_impurity import gini_impurity
from skrough.utils.mixin import DecTableOpsMixin
from skrough.base import Bireduct
import pandas.core.sorting
import pandas.core


class DynamicallyAdaptedApproximateBireduct(RoughBase, DecTableOpsMixin):

    def __init__(self, n_of_probes, allowed_randomness, candidate_n_attrs=None, max_n_attrs=None,
                 score_func=gini_impurity, random_state=None):
        super(DynamicallyAdaptedApproximateBireduct, self).__init__(score_func, random_state=random_state)
        self.candidate_n_attrs = candidate_n_attrs
        self.max_n_attrs = max_n_attrs
        self.n_of_probes = n_of_probes
        self.allowed_randomness = allowed_randomness

    def fit(self, x, y, sample_weight=None, check_data_consistency=True):
        super(DynamicallyAdaptedApproximateBireduct, self).fit(
            x, y,
            sample_weight=sample_weight,
            check_data_consistency=check_data_consistency
            )
        return self

    def get_best_attr(self, group_index, n_groups, candidate_attrs, x, x_count_distinct, y, y_count_distinct):
        scores = np.fromiter((
            self.split_groups_and_compute_chaos_score(group_index, n_groups, i, x, x_count_distinct, y,
                                                      y_count_distinct)
            for i in candidate_attrs), dtype=float)
        return candidate_attrs[scores.argmin()]

    def get_bireduct(self):
        # TODO: check this - is it needed?
        sklearn.utils.validation.check_is_fitted(self, ['x', 'y'])

        # TODO: introduce random_state usage
        xx = self.x
        xx_count_distinct = self.x_count_distinct
        yy = self.y
        yy_count_distinct = self.y_count_distinct

        if not self.check_data_consistency:
            total_chaos_score = 0

        group_index = np.zeros(len(xx), dtype=np.int_)
        n_groups = 1
        decision_chaos_score = self.compute_chaos_score(group_index, n_groups, xx, yy, yy_count_distinct)

        result_attrs = []
        while True:
            candidate_attrs = np.delete(np.arange(xx.shape[1]), result_attrs)
            # TODO: add inconsistent data handling
            if len(candidate_attrs) == 0:
                break
            if self.candidate_n_attrs is not None:
                # TODO: introduce random_state usage
                candidate_attrs = np.random.choice(candidate_attrs,
                                                   np.min([len(candidate_attrs), self.candidate_n_attrs]),
                                                   replace=False)
            best_attr = self.get_best_attr(group_index, n_groups, candidate_attrs, xx, xx_count_distinct, yy,
                                           yy_count_distinct)

            ###############################################
            # test the loop should stop - using attr probes
            ###############################################
            best_attr_values = xx[:, best_attr]
            best_attr_chaos_score = self.split_groups_and_compute_chaos_score_2(group_index, n_groups,
                                                                                best_attr_values, xx_count_distinct[best_attr],
                                                                                yy, yy_count_distinct)
            attr_is_better_count = 0
            for i in range(self.n_of_probes):
                best_attr_shuffled_values = np.random.permutation(best_attr_values)
                best_attr_shuffled_chaos_score = self.split_groups_and_compute_chaos_score_2(group_index, n_groups,
                                                                                             best_attr_shuffled_values, xx_count_distinct[best_attr],
                                                                                             yy, yy_count_distinct)
                attr_is_better_count += int(best_attr_chaos_score < best_attr_shuffled_chaos_score)

            best_attr_probe_score = (attr_is_better_count + 1) / (self.n_of_probes + 2)

            if best_attr_probe_score < (1 - self.allowed_randomness):
                if len(result_attrs) == 0:
                    result_attrs.append(int(best_attr))
                break
            ###############################################
            ###############################################

            result_attrs.append(int(best_attr))
            group_index, n_groups = split_groups(group_index,
                                                 n_groups,
                                                 xx[:, best_attr],
                                                 xx_count_distinct[best_attr],
                                                 compress_group_index=True)
            if self.max_n_attrs is not None and len(result_attrs) >= self.max_n_attrs:
                break

        # reduction phase
        before_reduction_chaos_score = self.compute_chaos_score(group_index, n_groups, xx, yy,
                                                           yy_count_distinct)
        result_attrs_reduction = list(result_attrs)
        # print(len(result_attrs))
        if len(result_attrs) > 1:
            for i in reversed(result_attrs):
                attrs_to_try = list(result_attrs_reduction)
                attrs_to_try.remove(i)

                group_index_reduction = pandas.core.sorting.get_group_index(xx[:, attrs_to_try].T,
                                                                            xx_count_distinct[attrs_to_try], sort=False,
                                                                            xnull=False)
                group_index_reduction, _ = pandas.core.sorting.compress_group_index(group_index_reduction, sort=False)
                n_groups_reduction = max(group_index_reduction) + 1

                current_chaos_score = self.compute_chaos_score(group_index_reduction, n_groups_reduction, xx, yy,
                                                               yy_count_distinct)
                if current_chaos_score <= before_reduction_chaos_score:
                    result_attrs_reduction = attrs_to_try

        # print(len(result_attrs_reduction))
        # print('----')

        # update group_index
        result_attrs = result_attrs_reduction
        group_index = pandas.core.sorting.get_group_index(xx[:, result_attrs].T,
                                                          xx_count_distinct[result_attrs], sort=False,
                                                          xnull=False)
        group_index, _ = pandas.core.sorting.compress_group_index(group_index, sort=False)
        n_groups = max(group_index) + 1

        # draw objects
        result_objects = draw_objects(group_index, yy)

        return Bireduct(result_objects, result_attrs)


def shrink(x, y, result_attrs):
    xx = x[:, result_attrs]
    return result_attrs
