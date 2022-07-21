import cython.parallel
import numpy as np

cimport cython
from libc.stdlib cimport abort, free, malloc


def gini_impurity_cython_1(distribution):
    # counts2 = np.zeros(distribution.shape[0], dtype=np.int_)
    # group_counts = np.zeros(distribution.shape[0], dtype=np.int_)
    return gini_impurity_cython_1_(distribution, distribution.shape[0], distribution.shape[1])

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double gini_impurity_cython_1_(long [:, :] distribution, Py_ssize_t nrow, Py_ssize_t ncol):
    cdef long * counts2 = <long *> malloc(sizeof(long) * nrow)
    cdef long * group_counts = <long *> malloc(sizeof(long) * nrow)
    cdef int i, j;
    cdef long x;
    cdef double result = 0.0
    for i in cython.parallel.prange(nrow, nogil=True):
        counts2[i] = 0
        group_counts[i] = 0
        for j in range(ncol):
            x = distribution[i, j]
            counts2[i] += x * x
            group_counts[i] += x
    for i in cython.parallel.prange(nrow, nogil=True):
        if group_counts[i] > 0:
            result += ((group_counts[i]*group_counts[i]) - counts2[i]) / group_counts[i]
    result = result / nrow
    free(counts2)
    free(group_counts)
    return result
