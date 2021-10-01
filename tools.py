# %%
import joblib
import more_itertools
import numba
import numpy as np
import pandas as pd
import pandas.core.sorting
import sklearn.utils
import timeit

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


# def _split_groups(group_index, n_groups, factorized_values, values_count_distinct, compress_group_index=True):
#     '''
#     Split groups of objects into finer groups according to values on a single splitting attribute
#     '''
#     group_index = group_index * values_count_distinct + factorized_values
#     if compress_group_index:
#         group_index, n_groups = pandas.core.sorting.compress_group_index(
#             group_index, sort=False)
#         n_groups = len(n_groups)
#     else:
#         n_groups = n_groups * values_count_distinct
#     return group_index, n_groups


# def split_into_groups(xx, xx_count_distinct, attrs):
#     '''
#     Split objects into groups according to values on given attributes
#     '''
#     group_index = np.zeros(len(xx), dtype=np.int_)
#     n_groups = 1
#     for attr in attrs:
#         group_index, n_groups = _split_groups(group_index,
#                                               n_groups,
#                                               xx[:, attr],
#                                               xx_count_distinct[attr],
#                                               compress_group_index=True)
#     return group_index, n_groups


def _split_into_groups(xx, xx_count_distinct, attrs):
    '''
    Split objects into groups according to values on given attributes
    '''
    if not attrs:
        group_index, n_groups = np.zeros(len(xx), dtype=np.int_), 1
    else:
        attrs = list(attrs)
        group_index = pandas.core.sorting.get_group_index(labels=x[:, attrs].T,
                                                          shape=x_count_distinct[attrs],
                                                          sort=False,
                                                          xnull=False)
        group_index, n_groups = pandas.core.sorting.compress_group_index(group_index=group_index,
                                                                         sort=False)
        n_groups = len(n_groups)
    return group_index, n_groups


def get_chaos_score(xx, xx_count_distinct, yy, yy_count_distinct, attrs, chaos_fun,
                    _split_into_groups_fun=_split_into_groups,
                    _compute_chaos_score_fun=_compute_chaos_score):
    '''
    Compute chaos score for the grouping (equivalence classes) induced by the given subset of attributes
    '''
    group_index, n_groups = _split_into_groups_fun(
        xx, xx_count_distinct, attrs)
    result = _compute_chaos_score_fun(
        group_index, n_groups, xx, yy, yy_count_distinct, chaos_fun)
    return result


# def get_feature_importance(xx, xx_count_distinct, yy, yy_count_distinct,
#                            column_names, reduct_list, chaos_fun,
#                            _get_chaos_score_fun=get_chaos_score):
#     assert xx.shape[1] == len(column_names)

#     counts = np.zeros(xx.shape[1])
#     total_gain = np.zeros(xx.shape[1])
#     for reduct in reduct_list:
#         reduct = list(reduct)
#         reduct_all_attrs = set(reduct)
#         starting_chaos_score = _get_chaos_score_fun(xx, xx_count_distinct,
#                                                     yy, yy_count_distinct,
#                                                     reduct_all_attrs, chaos_fun)
#         counts[reduct] += 1
#         for attr in reduct:
#             attrs_to_check = reduct_all_attrs.difference([attr])
#             current_chaos_score = _get_chaos_score_fun(xx, xx_count_distinct, yy, yy_count_distinct,
#                                                        attrs_to_check, chaos_fun)
#             score_gain = current_chaos_score - starting_chaos_score
#             total_gain[attr] += score_gain
#     avg_gain = np.divide(total_gain, counts, out=np.zeros_like(
#         total_gain), where=counts > 0)
#     result = pd.DataFrame({'column': column_names,
#                            'count': counts,
#                            'total_gain': total_gain,
#                            'avg_gain': avg_gain,
#                            })
#     return result


def _compute_reduct_score_gains(xx, xx_count_distinct, yy, yy_count_distinct, reduct, chaos_fun, _get_chaos_score_fun):
    '''
    Compute feature importance for a single reduct
    '''
    score_gains = {}
    reduct = list(reduct)
    reduct_all_attrs = set(reduct)
    starting_chaos_score = _get_chaos_score_fun(xx, xx_count_distinct,
                                                yy, yy_count_distinct,
                                                reduct_all_attrs, chaos_fun)
    for attr in reduct:
        attrs_to_check = reduct_all_attrs.difference([attr])
        current_chaos_score = _get_chaos_score_fun(xx, xx_count_distinct, yy, yy_count_distinct,
                                                   attrs_to_check, chaos_fun)
        score_gains[attr] = current_chaos_score - starting_chaos_score
    return score_gains


def get_feature_importance(xx, xx_count_distinct, yy, yy_count_distinct,
                           column_names, reduct_list, chaos_fun,
                           n_jobs=None,
                           _get_chaos_score_fun=get_chaos_score):
    '''
    Compute feature importance for a given collection of reducts
    '''
    assert xx.shape[1] == len(column_names)

    score_gains_list = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(
        _compute_reduct_score_gains)(xx, xx_count_distinct,
                                     yy, yy_count_distinct,
                                     reduct, chaos_fun,
                                     _get_chaos_score_fun=_get_chaos_score_fun) for reduct in reduct_list)

    counts = np.zeros(xx.shape[1])
    total_gain = np.zeros(xx.shape[1])
    for reduct, score_gains in zip(reduct_list, score_gains_list):
        reduct = list(reduct)
        counts[reduct] += 1
        for attr in reduct:
            total_gain[attr] += score_gains[attr]
    avg_gain = np.divide(total_gain, counts, out=np.zeros_like(
        total_gain), where=counts > 0)
    result = pd.DataFrame({'column': column_names,
                           'count': counts,
                           'total_gain': total_gain,
                           'avg_gain': avg_gain,
                           })
    return result


def _prepare_values(values):
    '''
    Prepare/factorize values
    '''
    factorized_values, uniques = pd.factorize(values, na_sentinel=None)
    uniques = len(uniques)
    return factorized_values, uniques


def prepare_df(df, target_column):
    '''
    Prepare/factorize data table
    '''
    y = df[target_column]
    x = df.drop(columns=target_column)
    data = x.apply(_prepare_values, 0)
    x = np.vstack(data.values[0]).T
    x_count_distinct = data.values[1].astype(int)
    y, y_count_distinct = _prepare_values(y)
    x, y = sklearn.utils.check_X_y(x, y, multi_output=False)
    return x, x_count_distinct, y, y_count_distinct


# %%
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
    target_column = 'Play'
    x, x_count_distinct, y, y_count_distinct = prepare_df(df, 'Play')
    column_names = np.array(
        [col for col in df.columns if col != target_column])

    print(df)
    print()

    for attrs in [[0], [0, 1], [0, 1, 3]]:
        for chaos_function in [gini_impurity, entropy]:
            print(f'chaos score for attrs {attrs}({column_names[attrs]}) '
                  f'using `{chaos_function.__name__}` chaos function = '
                  f'{get_chaos_score(x, x_count_distinct, y, y_count_distinct, attrs, chaos_fun=chaos_function)}'
                  )

    inputs_collection = [
        [[0, 2], [0, 3], [0], [2, 3], [1, 2, 3]],
        [[0], [0, 1], [1, 2]],
        list(more_itertools.powerset(range(4)))
    ]
    for input in inputs_collection:
        for chaos_function in [gini_impurity, entropy]:
            print(f'\nfeature importance computed using `{chaos_function.__name__}` chaos function for attribute set list: {input}')
            print(get_feature_importance(x, x_count_distinct, y, y_count_distinct,
                                        column_names, input,
                                        chaos_fun=chaos_function))

# %%
