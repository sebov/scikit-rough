''' Bireducts '''

import itertools
import numpy as np
import pandas as pd
import sklearn
import logging

from skrough.base import RoughBase
from skrough.utils.group_index import split_groups, draw_objects

from skrough.metrics.gini_impurity import gini_impurity


def generate_permutation(n_objects, n_attrs, attrs_weight=None, random_state=None):
    if attrs_weight is None:
        attrs_weight = float(n_objects) / n_attrs if n_attrs else 1
    weights = list(itertools.chain(
                itertools.repeat(1, n_objects),
                itertools.repeat(attrs_weight, n_attrs)
                ))
    result = pd.Series(range(len(weights)))
    result = list(result.sample(len(result), weights=weights, random_state=random_state))
    return result



class OrderingHeuristicBireduct(RoughBase):

    def __init__(self, epsilon=0.0, score_func=gini_impurity, random_state=None):
        super(OrderingHeuristicBireduct, self).__init__(score_func, random_state=random_state)
        self.epsilon = epsilon

    def fit(self, x, y, sample_weight=None, check_data_consistency=True):
        super(OrderingHeuristicBireduct, self).fit(
            x, y,
            sample_weight=sample_weight,
            check_data_consistency=check_data_consistency
        )
        if not 0 <= self.epsilon <= 1:
            raise ValueError(f'epsilon must be in (0, 1], got {self.epsilon}')

        return self

    def get_bireduct(self, random_state=None):
        sklearn.utils.validation.check_is_fitted(self, ['x', 'y'])

        group_index = np.zeros(len(self.x), dtype=np.int_)
        n_groups = 1

        result_attrs = []
        result_objects = []
        # permutation = generate_permutation(self.x.shape[0], self.x.shape[1], random_state=random_state)
        # for i in permutation:
        #     candidate_attrs = np.delete(np.arange(self.x.shape[1]), result_attrs)
        #     if self.n_attrs is not None:
        #         candidate_attrs = np.random.choice(candidate_attrs,
        #                                            np.min([len(candidate_attrs), self.n_attrs]),
        #                                            replace=False)
        #     best_attr = self.get_best_attr(group_index, n_groups, candidate_attrs)
        #     result_attrs.append(best_attr)
        #     group_index, n_groups = split_groups(group_index,
        #                                          n_groups,
        #                                          self.x[:, best_attr],
        #                                          self.x_count_distinct[best_attr],
        #                                          compress_group_index=True)
        #
        # group_index = np.zeros(len(self.x), dtype=np.int_)
        # n_groups = 1
        # for attr in result_attrs:
        #     group_index, n_groups = split_groups(group_index,
        #                                               n_groups,
        #                                               self.x[:, attr],
        #                                               self.x_count_distinct[attr],
        #                                               compress_group_index=True)
        #
        # result_objects = draw_objects(group_index, self.y)
        return result_attrs, result_objects
