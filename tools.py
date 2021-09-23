import numba
import numpy as np
import pandas as pd
import sklearn.utils
import pandas.core.sorting


@numba.njit
def _compute_dec_distribution(group_index, n_groups, factorized_dec_values, dec_values_count_distinct):
    '''
    Compute decision distribution within groups of objects
    '''
    result = np.zeros((n_groups, dec_values_count_distinct), dtype=np.int_)
    nrow = group_index.shape[0]
    for i in range(nrow):
        result[group_index[i], factorized_dec_values[i]] += 1
    return result


def _split_groups(group_index, n_groups, factorized_values, values_count_distinct, compress_group_index=True):
    '''
    Split groups of objects into finer groups according to values on a splitting attribute
    '''
    group_index = group_index * values_count_distinct + factorized_values
    if compress_group_index:
        group_index, n_groups = pandas.core.sorting.compress_group_index(
            group_index, sort=False)
        n_groups = len(n_groups)
    else:
        n_groups = n_groups * values_count_distinct
    return group_index, n_groups


@numba.njit
def gini_impurity(distribution: np.array, n: int):
    '''
    Compute average gini impurity

    Compute average gini impurity using the following formula

    .. math:: \sum((1 - \sum(counts^2)/(\sum(counts)^2)) * \sum(counts)) / n

    where counts correspond to rows in distribution
    '''
    nrow = distribution.shape[0]
    ncol = distribution.shape[1]
    result = 0.0
    for i in numba.prange(nrow):
        group_count = 0
        sum_squared_counts = 0
        for j in range(ncol):
            x = distribution[i, j]
            group_count += x
            sum_squared_counts += x * x
        if group_count > 0:
            result += (1.0 - sum_squared_counts /
                       (group_count * group_count)) * (group_count / n)
    return result


@numba.njit
def entropy(distribution: np.array, n: int):
    '''
    Compute average entropy
    '''
    nrow = distribution.shape[0]
    ncol = distribution.shape[1]
    result = 0.0
    for i in numba.prange(nrow):
        group_count = 0
        for j in range(ncol):
            group_count += distribution[i, j]
        if group_count > 0:
            tmp = 0.0
            for j in range(ncol):
                if distribution[i, j] > 0:
                    p = distribution[i, j] / group_count
                    tmp -= p * np.log2(p)
            result += tmp * (group_count / n)
    return result


def _compute_chaos_score(group_index, n_groups, xx, yy, yy_count_distinct, chaos_fun):
    '''
    Compute chaos score for the given grouping of objects (into equivalence classes)
    '''
    distribution = _compute_dec_distribution(
        group_index, n_groups, yy, yy_count_distinct)
    return chaos_fun(distribution, len(xx))


def get_chaos_score(xx, xx_count_distinct, yy, yy_count_distinct, attrs, chaos_fun):
    '''
    Compute chaos score for the grouping (equivalence classes) induced by the given subset of attributes
    '''
    group_index = np.zeros(len(xx), dtype=np.int_)
    n_groups = 1
    for attr in attrs:
        group_index, n_groups = _split_groups(group_index,
                                              n_groups,
                                              xx[:, attr],
                                              xx_count_distinct[attr],
                                              compress_group_index=True)
    result = _compute_chaos_score(
        group_index, n_groups, xx, yy, yy_count_distinct, chaos_fun)
    return result


def prepare(df, target_column):
    df = df.astype('category')
    df = df.apply(lambda x: x.cat.codes)
    df_dec = df.pop(target_column)
    x_count_distinct = df.nunique().values
    y_count_distinct = df_dec.nunique()
    x, y = sklearn.utils.check_X_y(df, df_dec)
    return x, x_count_distinct, y, y_count_distinct


if __name__ == '__main__':
    df = pd.DataFrame(np.array(
        [['sunny', 'hot', 'high', 'weak', 'no'],
         ['sunny', 'hot', 'high', 'strong', 'no'],
         ['overcast', 'hot', 'high', 'weak', 'yes'],
         ['rain', 'mild', 'high', 'weak', 'yes'],
         ['rain', 'cool', 'normal', 'weak', 'yes'],
         ['rain', 'cool', 'normal', 'strong', 'no'],
         ['overcast', 'cool', 'normal', 'strong', 'yes'],
         ['sunny', 'mild', 'high', 'weak', 'no'],
         ['sunny', 'cool', 'normal', 'weak', 'yes'],
         ['rain', 'mild', 'normal', 'weak', 'yes'],
         ['sunny', 'mild', 'normal', 'strong', 'yes'],
         ['overcast', 'mild', 'high', 'strong', 'yes'],
         ['overcast', 'hot', 'normal', 'weak', 'yes'],
         ['rain', 'mild', 'high', 'strong', 'no']], dtype=object),
        columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'Play'])
    x, x_count_distinct, y, y_count_distinct = prepare(df, 'Play')

    for attrs in [[0], [0, 1], [0, 1, 3]]:
        print(f'chaos score for attrs {attrs} using gini_impurity chaos function = '
              f'{get_chaos_score(x, x_count_distinct, y, y_count_distinct, attrs, chaos_fun=gini_impurity)}'
              )
        print(f'chaos score for attrs {attrs} using entropy chaos function = '
              f'{get_chaos_score(x, x_count_distinct, y, y_count_distinct, attrs, chaos_fun=entropy)}'
              )
