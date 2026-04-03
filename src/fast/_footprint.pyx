# cython: language_level=3
import numpy as np
cimport numpy as cnp
from libc.math cimport floor

cnp.import_array()


def footprint_histogram(
    cnp.ndarray[cnp.float64_t, ndim=1] price,
    cnp.ndarray[cnp.float64_t, ndim=1] signed_size,
    double tick_size,
    double price_min,
    double price_max,
):
    """
    Гистограмма футпринта по ценовым бинам (tick_size).
    signed_size > 0 агрессивные покупки, < 0 продажи (задаётся снаружи).
    """
    cdef Py_ssize_t n = price.shape[0]
    cdef Py_ssize_t k
    cdef int nb = <int>floor((price_max - price_min) / tick_size) + 3
    if nb < 4:
        nb = 4
    cdef cnp.ndarray[cnp.float64_t, ndim=1] bid_v = np.zeros(nb, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] ask_v = np.zeros(nb, dtype=np.float64)
    cdef double p, s
    cdef int idx
    cdef double inv = 1.0 / tick_size

    for k in range(n):
        p = price[k]
        s = signed_size[k]
        if p < price_min or p > price_max:
            continue
        idx = <int>floor((p - price_min) * inv)
        if idx < 0:
            idx = 0
        if idx >= nb:
            idx = nb - 1
        if s >= 0:
            ask_v[idx] += s
        else:
            bid_v[idx] += -s

    return np.asarray(bid_v), np.asarray(ask_v), price_min, tick_size
