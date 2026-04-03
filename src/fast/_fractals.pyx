# cython: language_level=3
import numpy as np
cimport numpy as cnp
from libc.math cimport fabs

cnp.import_array()


def fractal_highs(cnp.ndarray[cnp.float64_t, ndim=1] high, Py_ssize_t left, Py_ssize_t right):
    """
    Fractal up: high[i] строго выше high[i-left..i-1] и high[i+1..i+right].
    """
    cdef Py_ssize_t n = high.shape[0]
    cdef Py_ssize_t i, j
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] out = np.zeros(n, dtype=np.uint8)
    cdef double hi
    cdef int ok
    for i in range(left, n - right):
        hi = high[i]
        ok = 1
        for j in range(1, left + 1):
            if hi <= high[i - j]:
                ok = 0
                break
        if ok == 0:
            continue
        for j in range(1, right + 1):
            if hi <= high[i + j]:
                ok = 0
                break
        if ok:
            out[i] = 1
    return np.asarray(out, dtype=bool)


def fractal_lows(cnp.ndarray[cnp.float64_t, ndim=1] low, Py_ssize_t left, Py_ssize_t right):
    cdef Py_ssize_t n = low.shape[0]
    cdef Py_ssize_t i, j
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] out = np.zeros(n, dtype=np.uint8)
    cdef double lo
    cdef int ok
    for i in range(left, n - right):
        lo = low[i]
        ok = 1
        for j in range(1, left + 1):
            if lo >= low[i - j]:
                ok = 0
                break
        if ok == 0:
            continue
        for j in range(1, right + 1):
            if lo >= low[i + j]:
                ok = 0
                break
        if ok:
            out[i] = 1
    return np.asarray(out, dtype=bool)
