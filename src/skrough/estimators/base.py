# import numpy as np
# import pandas as pd
# import sklearn

# class RoughBase(sklearn.base.BaseEstimator):
#     def __init__(self, score_func, random_state):
#         # set initial state
#         self.x = None
#         self.x_count_distinct = None
#         self.y = None
#         self.y_count_distinct = None
#         self.check_data_consistency = None

#         self.score_func = score_func
#         self.random_state = random_state

#     def fit(self, x, y, sample_weight=None, check_data_consistency=False):
#         # self.check_data_consistency = check_data_consistency
#         if check_data_consistency:
#             # TODO: by default check_data_consistency should be True but for now we
#             # do not handle this
#             raise NotImplementedError("check_data_consistency==True not implemented")
#         else:
#             pass

#         # TODO: check if data is categorical
#         self.x, self.y = sklearn.utils.check_X_y(x, y, multi_output=False)
#         data = np.apply_along_axis(self.__prepare_values, 0, self.x)
#         self.x = np.vstack(data[0]).T
#         self.x_count_distinct = data[1]
#         self.y, self.y_count_distinct = self.__prepare_values(self.y)
#         return self

#     @staticmethod
#     def __prepare_values(values):
#         factorized_values, uniques = pd.factorize(values)
#         uniques = len(uniques)
#         return factorized_values, uniques
