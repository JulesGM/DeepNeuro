"""
spherical func very strongly inspired by http://stackoverflow.com/a/4116803/447599

we should probably do this in tf/theano...
"""

cdef extern from "math.h":
    long double sqrt(long double xx)
    long double atan2(long double a, double b)
    long double cos(long double a)
    long double sin(long double a)

import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float32_t DTYPE_t
py_DTYPE_t = np.float32

cdef DTYPE_t to_deg = 1.


@cython.boundscheck(False)
@cython.wraparound(False)
def cart_to_spherical(np.ndarray[DTYPE_t, ndim = 2] xyz):
    #cython & numpy array using mne.transforms._cartesian_to_sphere analogue
    cdef np.ndarray[DTYPE_t, ndim = 2] pts = np.empty((xyz.shape[0], 3), dtype=py_DTYPE_t)
    cdef DTYPE_t XsqPlusYsq
    for i in xrange(xyz.shape[0]):
        XsqPlusYsq = xyz[i,0]**2 + xyz[i, 1]**2
        pts[i, 0] = atan2(xyz[i, 2], sqrt(XsqPlusYsq)) * to_deg
        pts[i, 1] = atan2(xyz[i, 1], xyz[i, 0]) * to_deg
        pts[i, 2] = sqrt(XsqPlusYsq + xyz[i, 2]**2)
    return pts


@cython.boundscheck(False)
@cython.wraparound(False)
def cart_to_polar(np.ndarray[DTYPE_t, ndim = 2] xy):
    #there is no such thing as a mne.transforms._cartesian_to_polar function
    cdef np.ndarray[DTYPE_t, ndim = 2] pts = np.empty((xy.shape[0], 2), dtype=py_DTYPE_t)
    cdef DTYPE_t XsqPlusYsq
    for i in xrange(xy.shape[0]):
        pts[i, 0] = atan2(xy[i, 1], xy[i, 0]) * to_deg
        pts[i, 1] = sqrt(xy[i, 0]**2 + xy[i, 1]**2)
    return pts


@cython.boundscheck(False)
@cython.wraparound(False)
def polar_to_cart(np.ndarray[DTYPE_t, ndim = 2] theta_r):
    #there is no such thing as a mne.transforms._cartesian_to_polar function
    cdef np.ndarray[DTYPE_t,ndim = 2] pts = np.empty((theta_r.shape[0], 2), dtype=py_DTYPE_t)

    for i in xrange(theta_r.shape[0]):
        pts[i, 0] = theta_r[i, 1] * cos(theta_r[i, 0])
        pts[i, 1] = theta_r[i, 1] * sin(theta_r[i, 0])
    return pts