import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef compute(int[:] s1, int[:] s2):
    cdef Py_ssize_t sz1 = s1.shape[0]
    cdef Py_ssize_t sz2 = s2.shape[0]

    dist_np = np.zeros((sz1 + 1, sz2 + 1), dtype=np.intc)
    cdef int[:, :] dist = dist_np
    cdef Py_ssize_t i, j

    for i in range(sz1 + 1):
        dist[i, 0] = i
    for j in range(sz2 + 1):
        dist[0, j] = j

    for i in range(1, sz1 + 1):
        for j in range(1, sz2 + 1):
            dist[i, j] = min(dist[i, j - 1] + 1, dist[i - 1, j] + 1, dist[i - 1, j - 1] + (s1[i - 1] != s2[j - 1]))

    return dist[sz1, sz2]


def levenshtein(np.ndarray s1, np.ndarray s2):
    return compute(s1.astype(np.intc), s2.astype(np.intc))
