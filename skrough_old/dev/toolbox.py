import numpy as np
import pandas as pd
import skrough.reducts.greedy_heuristic_reduct
import skrough.bireducts.sampling_heuristic_bireduct

import sklearn
import sklearn.tree

sklearn.tree.Deci

df1 = pd.DataFrame([
    ['sunny', 'hot', 'high', 'weak', 'no'],
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
    ['rain', 'mild', 'high', 'strong', 'no'],
    ],
    columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'Play'])
df1_dec = df1.pop('Play')
df1 = df1.astype('category')
df1 = df1.apply(lambda x: x.cat.codes)

df2 = pd.read_csv('__path__/train_utf.csv', sep=';')
df2_dec = df2.pop('target')
df2 = df2.astype('category')
df2 = df2.apply(lambda x: x.cat.codes)


df, df_dec = df2, df2_dec


# ghr = skrough.reducts.greedy_heuristic_reduct.GreedyHeuristicReduct(n_candidate_attrs=100, epsilon=0.3)
# ghr.fit(df, df_dec, check_data_consistency=False)
# red = ghr.get_reduct()
# print(red)

# shr = skrough.bireducts.sampling_heuristic_bireduct.SamplingHeuristicBireduct(n_attrs=20)
# shr.fit(df, df_dec, check_data_consistency=False)
# bir = shr.get_bireduct()




from collections.abc import Iterable
from skrough.base import Reduct, Bireduct

def test_functional_dependency(x, y, objects=None, attributes=None):
    objects = objects if objects is not None else slice(None)
    attributes = attributes if attributes is not None else slice(None)
    if isinstance(objects, Iterable) and isinstance(attributes, Iterable):
        x_index = np.ix_(objects, attributes)
    else:
        x_index = np.index_exp[objects, attributes]
    dfx = pd.DataFrame(x[x_index])
    dfy = pd.DataFrame(y[objects])
    df = pd.concat([dfx, dfy], axis=1)
    if df.shape[0] == 0:
        duplicated = 0
    elif df.shape[1] == 0:
        duplicated = df.shape[0]
    else:
        duplicated = df.iloc[:, :-1].duplicated().sum()
    duplicated_with_dec = df.duplicated().sum()
    return duplicated == duplicated_with_dec

def test_if_reduct(x, y, red):
    # TODO: what if red does not hold functional dependency?
    for i in red.attributes:
        attributes = np.setdiff1d(red.attributes, [i])
        if test_functional_dependency(x, y, attributes=attributes):
            return False
    return True

def test_if_bireduct(x, y, bir):
    xx = x[np.ix_(bir.objects, bir.attributes)]
    yy = y[bir.objects]
    if not test_if_reduct(xx, yy, Reduct(bir.attributes)):
        return False
    else:
        return True

print(test_if_bireduct(df, df_dec, bir))


# import time
# start = time.time()
# bireducts = []
# N = 100
# for i in range(N):
#     print(f'{i}/{N}')
#     bireducts.append(shr.get_bireduct())
# end = time.time()
# print(f'elapsed {end-start} seconds')