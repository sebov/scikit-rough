# import logging

# import sklearn.utils

# # ------------------------------------------------------------------


# def get_bireduct(df, dec, permutation):
#     """
#     For a given columns-objects permutation compute a bireduct using
#     the ordering algorithm.
#     """
#     orig_cols = list(df.columns[df.columns != dec])
#     orig_rows = list(df.index)
#     cols = set(orig_cols)
#     rows = set()
#     for i, p in enumerate(permutation):
#         if i % 50 == 0:
#             logging.debug(f"{i}/{len(permutation)}")
#         if p < len(df):
#             new_rows = rows | {orig_rows[p]}
#             if check_functional_dependency(df, dec, list(cols), list(new_rows)):
#                 rows = new_rows
#         else:
#             new_cols = cols - {orig_cols[p - len(df)]}
#             if check_functional_dependency(df, dec, list(new_cols), list(rows)):
#                 cols = new_cols
#     return (tuple(rows), tuple(cols))


# def get_random_bireduct(df, dec, ratio, random_state=None):
#     """
#     Generate random bireduct with a given 'ratio' influencing the shuffling
#     of columns and objects.
#     """
#     random_state = sklearn.utils.check_random_state(random_state)
#     attrs_weight = float(ratio) * 2 * df.shape[0] / df.shape[1]
#     permutation = generate_permutation(df, dec, attrs_weight,
# random_state=random_state)
#     return get_bireduct(df, dec, permutation)


# def get_bireducts_for_ratio(df, dec, ratio, n_bireducts, random_state=None):
#     random_state = sklearn.utils.check_random_state(random_state)
#     result = []
#     for i in range(n_bireducts):
#         if i % 50 == 0:
#             logging.debug(f"bireduct {i}/{n_bireducts}")
#         result.append(get_random_bireduct(df, dec, ratio, random_state))
#     return result
