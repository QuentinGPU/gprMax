# Copyright (C) 2015-2023: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
cimport numpy as np
from cython.parallel import prange

from gprMax.constants cimport floattype_t
from gprMax.constants cimport complextype_t
from scipy.constants import epsilon_0 as e0
from scipy.constants import mu_0 as mu0


##################
# Update borders #
##################
cpdef void update_border_X(
                    int nx,
                    int ny,
                    int nz,
                    int nthreads,
                    floattype_t[:, :, ::1] Ex,
                    floattype_t[:, :, ::1] Ey,
                    floattype_t[:, :, ::1] Ez,
                    floattype_t[:, :, ::1] Hx,
                    floattype_t[:, :, ::1] Hy,
                    floattype_t[:, :, ::1] Hz,
                    floattype_t phase,
            ):
    """This function updates the electric field components.

    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
    """

    cdef Py_ssize_t j, k
    for j in prange(0, ny, nogil= True, schedule= 'static', num_threads = nthreads):
        for k in range(0, nz):
            Ex[nx+1, j, k] = phase * Ex[nx-1, j, k]
            Ey[nx+1, j, k] = phase * Ey[nx-1, j, k]
            Ez[nx+1, j, k] = phase * Ez[nx-1, j, k]
            Hx[nx+1, j, k] = - phase * Hx[nx-1, j, k]
            Hy[nx+1, j, k] = - phase * Hy[nx-1, j, k]
            Hz[nx+1, j, k] = - phase * Hz[nx-1, j, k]

cpdef void update_border_Y(
                    int nx,
                    int ny,
                    int nz,
                    int nthreads,
                    floattype_t[:, :, ::1] Ex,
                    floattype_t[:, :, ::1] Ey,
                    floattype_t[:, :, ::1] Ez,
                    floattype_t[:, :, ::1] Hx,
                    floattype_t[:, :, ::1] Hy,
                    floattype_t[:, :, ::1] Hz,
                    floattype_t phase,
            ):
    """This function updates the electric field components.

    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
    """

    cdef Py_ssize_t i, k
    for i in prange(0, nx, nogil= True, schedule= 'static', num_threads = nthreads):
        for k in range(0, nz):
            Ex[i, ny+1, k] = phase * Ex[i, ny-1, k]
            Ey[i, ny+1, k] = phase * Ey[i, ny-1, k]
            Ez[i, ny+1, k] = phase * Ez[i, ny-1, k]
            Hx[i, ny+1, k] = - phase * Hx[i, ny-1, k]
            Hy[i, ny+1, k] = - phase * Hy[i, ny-1, k]
            Hz[i, ny+1, k] = - phase * Hz[i, ny-1, k]

cpdef void update_electric_border_Z(
                    int nx,
                    int ny,
                    int nz,
                    int nthreads,
                    floattype_t[:, :, ::1] Ex,
                    floattype_t[:, :, ::1] Ey,
                    floattype_t[:, :, ::1] Ez,
                    floattype_t[:, :, ::1] Hx,
                    floattype_t[:, :, ::1] Hy,
                    floattype_t[:, :, ::1] Hz,
                    floattype_t phase,
            ):
    """This function updates the electric field components.

    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
    """

    cdef Py_ssize_t i, j
    
    for j in prange(0, ny, nogil= True, schedule= 'static', num_threads = nthreads):
        for k in range(0, nz):
            Ex[i, j, nz+1] = phase * Ex[i, j, nz-1]
            Ey[i, j, nz+1] = phase * Ey[i, j, nz-1]
            Ez[i, j, nz+1] = phase * Ez[i, j, nz-1]
            Hx[i, j, nz+1] = - phase * Hx[i, j, nz-1]
            Hy[i, j, nz+1] = - phase * Hy[i, j, nz-1]
            Hz[i, j, nz+1] = - phase * Hz[i, j, nz-1]

