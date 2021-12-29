import numba
import numpy as np
import pandas.core

@numba.njit
def compute_dec_distribution_orig(group_index, n_groups, factorized_dec_values, dec_values_count_distinct):
    '''
    Compute decision distribution within groups of objects
    '''
    result = np.zeros((n_groups, dec_values_count_distinct))
    for group, dec in zip(group_index, factorized_dec_values):
        result[group, dec] += 1
    return result


@numba.njit()
def compute_dec_distribution(group_index, n_groups, factorized_dec_values, dec_values_count_distinct):
    '''
    Compute decision distribution within groups of objects
    '''
    result = np.zeros((n_groups, dec_values_count_distinct), dtype=np.int_)
    nrow = group_index.shape[0]
    for i in range(nrow):
        result[group_index[i], factorized_dec_values[i]] += 1
    return result



def compute_homogeneity(distribution):
    '''
    Compute decision homogeneity within groups of objects
    '''
    # check in which rows there are no more than one positive values
    return np.sum(distribution > 0, axis=1) <= 1


def split_groups(group_index, n_groups, factorized_values, values_count_distinct, compress_group_index=True):
    '''
    Split groups of objects into finer groups according to values on a splitting attribute
    '''
    group_index = group_index * values_count_distinct + factorized_values
    if compress_group_index:
        group_index, n_groups = pandas.core.sorting.compress_group_index(group_index, sort=False)
        n_groups = len(n_groups)
    else:
        n_groups = n_groups * values_count_distinct
    return group_index, n_groups

def draw_objects(group_index, dec_values, permutation=None):
    '''
    Draw objects having uniform decision values within their groups
    '''
    if permutation is None:
        permutation = np.random.permutation(len(group_index))
    _, idx = np.unique(group_index[permutation], return_index=True)
    idx = permutation[idx]
    group_dec_values = dict(zip(group_index[idx], dec_values[idx]))
    result = [i for i in range(len(group_index))
              if dec_values[i] == group_dec_values[group_index[i]]]
    return result

# def draw_objects_new(group_index, dec_values, permutation=None):
#     '''
#     Draw objects having uniform decision values within their groups
#     '''
#     tab = np.concatenate([group_index[:, np.newaxis], dec_values[:, np.newaxis]], axis=1)
#     if permutation is None:
#         permutation = np.random.permutation(tab.shape[0])
#     _, idx = np.unique(tab[permutation], return_index=True, axis=0)
#     idx = permutation[idx]
#     np.random.shuffle(idx)
#     group_dec_values = dict(tab[idx])
#     result = [i for i in range(len(group_index))
#               if dec_values[i] == group_dec_values[group_index[i]]]
#     return result