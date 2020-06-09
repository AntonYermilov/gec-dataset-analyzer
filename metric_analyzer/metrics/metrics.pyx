from typing import Tuple
import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef compute(int[:] s1, int[:] s2):
    cdef Py_ssize_t sz1 = s1.shape[0]
    cdef Py_ssize_t sz2 = s2.shape[0]

    cdef int S = 0
    cdef int I = 0
    cdef int D = 0

    cdef int counter = 1
    cdef int updated = 0

    dist_np = np.zeros((sz1 + 1, sz2 + 1), dtype=np.intc)
    cdef int[:, :] dist = dist_np

    cdef np.ndarray mask_s1 = np.zeros(sz1, dtype=np.intc)
    cdef np.ndarray mask_s2 = np.zeros(sz2, dtype=np.intc)

    cdef Py_ssize_t i, j

    for i in range(sz1 + 1):
        dist[i, 0] = i
    for j in range(sz2 + 1):
        dist[0, j] = j

    for i in range(1, sz1 + 1):
        for j in range(1, sz2 + 1):
            dist[i, j] = min(dist[i, j - 1] + 1, dist[i - 1, j] + 1, dist[i - 1, j - 1] + (s1[i - 1] != s2[j - 1]))

    i = sz1
    j = sz2

    while i > 0 and j > 0:
        if dist[i, j] == dist[i - 1, j - 1] + 1:
            i -= 1
            j -= 1
            S += 1
            mask_s1[i] = counter
            mask_s2[j] = counter
            updated = 1
            continue
        if dist[i, j] == dist[i - 1, j] + 1:
            i -= 1
            D += 1
            mask_s1[i] = counter
            updated = 1
            continue
        if dist[i, j] == dist[i, j - 1] + 1:
            j -= 1
            I += 1
            mask_s2[j] = counter
            updated = 1
            continue
        i -= 1
        j -= 1
        if updated == 1:
            counter += 1
            updated = 0

    D += i
    I += j

    assert S + I + D == dist[sz1, sz2]
    return S, I, D, mask_s1, mask_s2


def levenshtein(np.ndarray s1, np.ndarray s2) -> int:
    S, I, D, mask_s1, mask_s2 = compute(s1.astype(np.intc), s2.astype(np.intc))
    return S + I + D


def levenshtein_sid(np.ndarray s1, np.ndarray s2) -> Tuple[int, int, int, np.ndarray, np.ndarray]:
    S, I, D, mask_s1, mask_s2 = compute(s1.astype(np.intc), s2.astype(np.intc))
    return S, I, D, mask_s1, mask_s2
