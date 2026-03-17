# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np


def _demo1291_rec(double tea1, double cd, double[:] avec1, double[:] sig1):
    cdef int ncons = avec1.shape[0]
    cdef np.ndarray[double, ndim=1] xvec1 = np.empty(ncons, dtype=np.float64)
    cdef int i
    cdef double val

    for i in range(ncons):
        val = (avec1[i] - tea1) * sig1[i]
        if val > 0.0:
            xvec1[i] = val * sig1[i]
        else:
            xvec1[i] = 0.0

    cdef double r1 = tea1 / cd
    return xvec1, r1


def _demo1291_objd_batch(double[:] tea1s, double cd,
                         double[:] avec1, double[:] sig1):
    cdef int npts = tea1s.shape[0]
    cdef int ncons = avec1.shape[0]
    cdef np.ndarray[double, ndim=1] grad_arr = np.empty(npts, dtype=np.float64)
    cdef int p, i
    cdef double tea, val, s, xsum, r1

    for p in range(npts):
        tea = tea1s[p]
        xsum = 0.0
        for i in range(ncons):
            val = (avec1[i] - tea) * sig1[i]
            if val > 0.0:
                s = val * sig1[i]
            else:
                s = 0.0
            xsum += s
        r1 = tea / cd
        grad_arr[p] = xsum - r1

    return grad_arr


def solve_cqkp(double cd, double[:] avec1_mv, double[:] sigvec1_mv):
    cdef int ncons = avec1_mv.shape[0]
    cdef int i, j1

    cdef np.ndarray[double, ndim=1] avec1 = np.asarray(avec1_mv)
    cdef np.ndarray[double, ndim=1] sigvec1 = np.asarray(sigvec1_mv)

    cdef np.ndarray[double, ndim=1] advec1 = np.empty(ncons + 1, dtype=np.float64)
    advec1[0] = avec1[0] - 1.0
    advec1[ncons] = avec1[ncons - 1] + 1.0
    for i in range(1, ncons):
        advec1[i] = 0.5 * (avec1[i - 1] + avec1[i])

    grad1s_ep = _demo1291_objd_batch(avec1_mv, cd, avec1_mv, sigvec1_mv)

    j1 = ncons
    for i in range(ncons):
        if grad1s_ep[i] <= 0.0:
            j1 = i
            break

    cdef np.ndarray[double, ndim=1] l1 = np.zeros(ncons, dtype=np.float64)
    cdef double threshold = advec1[j1]
    for i in range(ncons):
        if (avec1[i] - threshold) * sigvec1[i] >= 0.0:
            l1[i] = 1.0

    cdef double sum_l1 = 0.0
    cdef double dot_al = 0.0
    for i in range(ncons):
        sum_l1 += l1[i]
        dot_al += avec1[i] * l1[i]
    cdef double tea_star = dot_al / (sum_l1 + 1.0 / cd)

    xvec1_star, _ = _demo1291_rec(tea_star, cd, avec1_mv, sigvec1_mv)
    return xvec1_star
