import numpy as np
from skrough.metrics.gini_impurity import gini_impurity_orig, gini_impurity
import pathlib
import pickle
import timeit


# distribution = np.random.randint(100, size=(1000000, 50))
# with pathlib.Path('/tmp/dist.pickle').open('wb') as f:
#     pickle.dump(distribution, file=f)

with pathlib.Path('/tmp/dist.pickle').open('rb') as f:
    distribution = pickle.load(f)

distribution = np.asarray(distribution)
n = distribution.sum()

print(timeit.timeit(lambda: gini_impurity_orig(distribution, n), number=1))
print(timeit.timeit(lambda: gini_impurity_orig(distribution, n), number=1))
print(timeit.timeit(lambda: gini_impurity(distribution, n), number=1))
print(timeit.timeit(lambda: gini_impurity(distribution, n), number=1))
print('--------------')
print(gini_impurity_orig(distribution, n))
print(gini_impurity(distribution, n))
