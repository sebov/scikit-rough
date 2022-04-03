# import numpy as np
# import skrough.typing as rght
# from skrough.structs import State

# def grow_shrink(
#     x: np.ndarray,
#     x_counts: np.ndarray,
#     y: np.ndarray,
#     y_count: int,
#     config: dict,
#     seed: rght.Seed = None,
# ):
#     rng = np.random.default_rng(seed)

#     state = State()


#     state = init_state(x, x_counts, y, y_count, config)

#     result_attrs = []

#     # init group_index
#     state["group_index"] = GroupIndex.create_one_group(len(x))

#     while True:

#         if check_stop_condition(x, x_counts, y, y_count, state, config):
#             break

#         candidate_attrs: np.ndarray = np.delete(
#             np.arange(x.shape[1]),
#             result_attrs,
#         )

#         attr = candidate_attrs[0]
#         state["group_index"] = state["group_index"].split(x[:, attr], x_counts[attr])
#         result_attrs.append(attr)


#     # check stop condition

#     # get candidates

#     # check if candidates can be add

#     # reduction phase

#     # prepare result

#     # compute total chaos score

#     result = prepare_result(x, x_counts, y, y_count, state, config)
#     return result
