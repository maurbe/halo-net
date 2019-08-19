from __future__ import print_function
cimport numpy as np
cimport cython

import numpy as no

@cython.boundscheck(False)
@cython.wraparound(False)
def CICDeposit_3(np.ndarray[np.float64_t, ndim=1] posx,
                 np.ndarray[np.float64_t, ndim=1] posy,
                 np.ndarray[np.float64_t, ndim=1] posz,
                 np.ndarray[np.float64_t, ndim=1] mass,
                 np.int64_t npositions,
                 np.ndarray[np.float64_t, ndim=3] field,
                 np.ndarray[np.float64_t, ndim=1] leftEdge,
                 np.ndarray[np.int32_t, ndim=1] gridDimension,
                 np.float64_t cellSize):

    cdef int i1, j1, k1, n
    cdef np.float64_t xpos, ypos, zpos
    cdef np.float64_t fact, edge0, edge1, edge2
    cdef np.float64_t le0, le1, le2
    cdef np.float64_t dx, dy, dz, dx2, dy2, dz2

    edge0 = (<np.float64_t> gridDimension[0]) - 0.5001
    edge1 = (<np.float64_t> gridDimension[1]) - 0.5001
    edge2 = (<np.float64_t> gridDimension[2]) - 0.5001
    fact = 1.0 / cellSize

    le0 = leftEdge[0]
    le1 = leftEdge[1]
    le2 = leftEdge[2]

    for n in range(npositions):
        #print(n, end=' ')
        #print()
        # Compute the position of the central cell
        xpos = (posx[n] - le0)*fact
        ypos = (posy[n] - le1)*fact
        zpos = (posz[n] - le2)*fact

        if (xpos < 0.5001) or (xpos > edge0):
            continue
        if (ypos < 0.5001) or (ypos > edge1):
            continue
        if (zpos < 0.5001) or (zpos > edge2):
            continue

        i1  = <int> (xpos + 0.5)
        j1  = <int> (ypos + 0.5)
        k1  = <int> (zpos + 0.5)

        # Compute the weights
        dx = (<np.float64_t> i1) + 0.5 - xpos
        dy = (<np.float64_t> j1) + 0.5 - ypos
        dz = (<np.float64_t> k1) + 0.5 - zpos
        dx2 =  1.0 - dx
        dy2 =  1.0 - dy
        dz2 =  1.0 - dz

        # Interpolate from field into sumfield
        field[i1-1,j1-1,k1-1] += mass[n] * dx  * dy  * dz
        field[i1  ,j1-1,k1-1] += mass[n] * dx2 * dy  * dz
        field[i1-1,j1  ,k1-1] += mass[n] * dx  * dy2 * dz
        field[i1  ,j1  ,k1-1] += mass[n] * dx2 * dy2 * dz
        field[i1-1,j1-1,k1  ] += mass[n] * dx  * dy  * dz2
        field[i1  ,j1-1,k1  ] += mass[n] * dx2 * dy  * dz2
        field[i1-1,j1  ,k1  ] += mass[n] * dx  * dy2 * dz2
        field[i1  ,j1  ,k1  ] += mass[n] * dx2 * dy2 * dz2
    return field


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def CICDeposit_3_weights_adjusted(np.ndarray[np.float64_t, ndim=1] posx,
                 np.ndarray[np.float64_t, ndim=1] posy,
                 np.ndarray[np.float64_t, ndim=1] posz,
                 np.ndarray[np.float64_t, ndim=1] mass,
                 np.int64_t npositions,
                 np.ndarray[np.float64_t, ndim=3] field,
                 np.ndarray[np.float64_t, ndim=3] weight_field,
                 np.ndarray[np.float64_t, ndim=1] leftEdge,
                 np.ndarray[np.int32_t, ndim=1] gridDimension,
                 np.float64_t cellSize):

    cdef int i1, j1, k1, n
    cdef np.float64_t xpos, ypos, zpos
    cdef np.float64_t fact, edge0, edge1, edge2
    cdef np.float64_t le0, le1, le2
    cdef np.float64_t dx, dy, dz, dx2, dy2, dz2

    edge0 = (<np.float64_t> gridDimension[0]) - 0.5001
    edge1 = (<np.float64_t> gridDimension[1]) - 0.5001
    edge2 = (<np.float64_t> gridDimension[2]) - 0.5001
    fact = 1.0 / cellSize

    le0 = leftEdge[0]
    le1 = leftEdge[1]
    le2 = leftEdge[2]

    for n in range(npositions):
        # Compute the position of the central cell
        xpos = (posx[n] - le0)*fact
        ypos = (posy[n] - le1)*fact
        zpos = (posz[n] - le2)*fact

        if (xpos < 0.5001) or (xpos > edge0):
            continue
        if (ypos < 0.5001) or (ypos > edge1):
            continue
        if (zpos < 0.5001) or (zpos > edge2):
            continue

        i1  = <int> (xpos + 0.5)
        j1  = <int> (ypos + 0.5)
        k1  = <int> (zpos + 0.5)

        # Compute the weights
        dx = (<np.float64_t> i1) + 0.5 - xpos
        dy = (<np.float64_t> j1) + 0.5 - ypos
        dz = (<np.float64_t> k1) + 0.5 - zpos
        dx2 =  1.0 - dx
        dy2 =  1.0 - dy
        dz2 =  1.0 - dz

        # Interpolate from field into sumfield
        field[i1-1,j1-1,k1-1] += mass[n] * dx  * dy  * dz
        field[i1  ,j1-1,k1-1] += mass[n] * dx2 * dy  * dz
        field[i1-1,j1  ,k1-1] += mass[n] * dx  * dy2 * dz
        field[i1  ,j1  ,k1-1] += mass[n] * dx2 * dy2 * dz
        field[i1-1,j1-1,k1  ] += mass[n] * dx  * dy  * dz2
        field[i1  ,j1-1,k1  ] += mass[n] * dx2 * dy  * dz2
        field[i1-1,j1  ,k1  ] += mass[n] * dx  * dy2 * dz2
        field[i1  ,j1  ,k1  ] += mass[n] * dx2 * dy2 * dz2

        weight_field[i1-1, j1-1, k1-1] += dx * dy * dz
        weight_field[i1  , j1-1, k1-1] += dx2 * dy * dz
        weight_field[i1-1, j1  , k1-1] += dx * dy2 * dz
        weight_field[i1  , j1  , k1-1] += dx2 * dy2 * dz
        weight_field[i1-1, j1-1, k1  ] += dx * dy * dz2
        weight_field[i1  , j1-1, k1  ] += dx2 * dy * dz2
        weight_field[i1-1, j1  , k1  ] += dx * dy2 * dz2
        weight_field[i1  , j1  , k1  ] += dx2 * dy2 * dz2

    return field / (weight_field+1e-10)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def masking(np.ndarray[np.float64_t, ndim=1] posx,
                 np.ndarray[np.float64_t, ndim=1] posy,
                 np.ndarray[np.float64_t, ndim=1] posz,
                 np.ndarray[np.float64_t, ndim=1] mass,
                 np.int64_t npositions,
                 np.ndarray[np.float64_t, ndim=3] field,
                 np.ndarray[np.float64_t, ndim=1] leftEdge,
                 np.ndarray[np.int32_t, ndim=1] gridDimension,
                 np.float64_t cellSize):

    cdef int i1, j1, k1, n
    cdef np.float64_t xpos, ypos, zpos
    cdef np.float64_t fact, edge0, edge1, edge2
    cdef np.float64_t le0, le1, le2
    cdef np.float64_t dx, dy, dz, dx2, dy2, dz2

    edge0 = (<np.float64_t> gridDimension[0]) - 0.5001
    edge1 = (<np.float64_t> gridDimension[1]) - 0.5001
    edge2 = (<np.float64_t> gridDimension[2]) - 0.5001
    fact = 1.0 / cellSize

    le0 = leftEdge[0]
    le1 = leftEdge[1]
    le2 = leftEdge[2]

    for n in range(npositions):
        # Compute the position of the central cell
        xpos = (posx[n] - le0)*fact
        ypos = (posy[n] - le1)*fact
        zpos = (posz[n] - le2)*fact

        if (xpos < 0.5001) or (xpos > edge0):
            continue
        if (ypos < 0.5001) or (ypos > edge1):
            continue
        if (zpos < 0.5001) or (zpos > edge2):
            continue

        i1  = <int> (xpos + 0.5)
        j1  = <int> (ypos + 0.5)
        k1  = <int> (zpos + 0.5)

        # Interpolate from field into sumfield
        field[i1-1,j1-1,k1-1] = mass[n]
        field[i1  ,j1-1,k1-1] = mass[n]
        field[i1-1,j1  ,k1-1] = mass[n]
        field[i1  ,j1  ,k1-1] = mass[n]
        field[i1-1,j1-1,k1  ] = mass[n]
        field[i1  ,j1-1,k1  ] = mass[n]
        field[i1-1,j1  ,k1  ] = mass[n]
        field[i1  ,j1  ,k1  ] = mass[n]

    return field
