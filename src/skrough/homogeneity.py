import numpy as np


def compute_homogeneity(distribution):
    """
    Compute decision homogeneity within groups of objects
    """
    # check in which rows there are no more than one positive values
    return np.sum(distribution > 0, axis=1) <= 1
