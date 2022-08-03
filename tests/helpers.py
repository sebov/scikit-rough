import numpy as np

INTEGERS_MAX = 20
testrng = np.random.default_rng()


def generate_data(size, values_max=INTEGERS_MAX):
    return testrng.integers(values_max, size=size)
