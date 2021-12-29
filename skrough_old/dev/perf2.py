import numpy as np
import pandas as pd
from skrough.utils.group_index import compute_dec_distribution, compute_dec_distribution_orig
import timeit
import pickle
import pathlib

# group_index = np.random.randint(500000, size=50000000)
# with pathlib.Path('/tmp/group_index.pickle').open('wb') as f:
#     pickle.dump(group_index, file=f)
# factorized_dec_values = np.random.randint(10, size=50000000)
# factorized_dec_values, _ = pd.factorize(factorized_dec_values)
# with pathlib.Path('/tmp/factorized_dec_values.pickle').open('wb') as f:
#     pickle.dump(factorized_dec_values, file=f)



with pathlib.Path('/tmp/group_index.pickle').open('rb') as f:
    group_index = pickle.load(f)
n_groups = group_index.max() + 1
with pathlib.Path('/tmp/factorized_dec_values.pickle').open('rb') as f:
    factorized_dec_values = pickle.load(f)
dec_values_count_distinct = len(np.unique(factorized_dec_values))

print(timeit.timeit(lambda: compute_dec_distribution_orig(group_index, n_groups, factorized_dec_values, dec_values_count_distinct), number=1))
print(timeit.timeit(lambda: compute_dec_distribution_orig(group_index, n_groups, factorized_dec_values, dec_values_count_distinct), number=1))
print(timeit.timeit(lambda: compute_dec_distribution(group_index, n_groups, factorized_dec_values, dec_values_count_distinct), number=1))
print(timeit.timeit(lambda: compute_dec_distribution(group_index, n_groups, factorized_dec_values, dec_values_count_distinct), number=1))
print('--------------')
x1 = compute_dec_distribution_orig(group_index, n_groups, factorized_dec_values, dec_values_count_distinct)
x2 = compute_dec_distribution(group_index, n_groups, factorized_dec_values, dec_values_count_distinct)
print(np.array_equal(x1, x2))
