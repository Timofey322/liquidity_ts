# cython: language_level=3
import numpy as np
cimport numpy as cnp

cnp.import_array()


def detect_fvg_zones(
    cnp.ndarray[cnp.float64_t, ndim=1] high,
    cnp.ndarray[cnp.float64_t, ndim=1] low,
    cnp.ndarray[cnp.float64_t, ndim=1] close,  # unused; единый API
):
    """
    FVG (имбаланс): bull low[i] > high[i-2], bear high[i] < low[i-2].
    Возвращает маски активной зоны на каждом баре (упрощённо: зона «жива» N баров с момента формирования).
    """
    cdef Py_ssize_t n = high.shape[0]
    cdef Py_ssize_t i
    cdef double h2, l2, hi, li
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] bull = np.zeros(n, dtype=np.uint8)
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] bear = np.zeros(n, dtype=np.uint8)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] z_lo = np.full(n, np.nan)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] z_hi = np.full(n, np.nan)

    for i in range(2, n):
        h2 = high[i - 2]
        l2 = low[i - 2]
        hi = high[i]
        li = low[i]
        if li > h2:
            bull[i] = 1
            z_lo[i] = h2
            z_hi[i] = li
        elif hi < l2:
            bear[i] = 1
            z_lo[i] = hi
            z_hi[i] = l2

    return (
        np.asarray(bull, dtype=bool),
        np.asarray(bear, dtype=bool),
        np.asarray(z_lo),
        np.asarray(z_hi),
    )
