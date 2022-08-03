import numpy as np

INTEGERS_MAX = 100
testrng = np.random.default_rng()


def generate_data(size):
    return testrng.integers(INTEGERS_MAX, size=size)
