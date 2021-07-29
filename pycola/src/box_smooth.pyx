########################################################################
########################################################################
#    Copyright (c) 2013,2014       Svetlin Tassev
#                       Princeton University,Harvard University
#
#   This file is part of pyCOLA.
#
#   pyCOLA is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   pyCOLA is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with pyCOLA.  If not, see <http://www.gnu.org/licenses/>.
#
########################################################################
########################################################################


import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.embedsignature(True)
def box_smooth(
                 np.ndarray[np.float32_t, ndim=3] arr,
                 np.ndarray[np.float32_t, ndim=3] arr1
                 ):
    """
    :math:`\\vspace{-1mm}`

    Do a 3x3x3 boxcar smoothing.
    
    **Arguments**:
    
    * ``arr`` -- a 3-dim float32 array serving as input.
    
    * ``arr1`` -- a 3-dim float32 array serving as output.
    
    .. note:: This should really be replaced with a Gaussian smoothing, 
       so that one can change the amount of smoothing. Gaussian 
       smoothing can be trivially implemented by modifying 
       :func:`potential.get_phi` as indicated in the source file of 
       that function. Not done here as this worked well enough for the 
       paper.
    
    """
    
    cdef int i,j,k
   
    
    cdef int ngrid_x,ngrid_y,ngrid_z
    
    ngrid_x=arr.shape[0]
    ngrid_y=arr.shape[1]
    ngrid_z=arr.shape[2]
    

    from cython.parallel cimport prange,parallel
    cdef int nthreads
    from multiprocessing import cpu_count
    nthreads=cpu_count()
    #print 'nthreads,npart_x = ',  nthreads,npart_x
    
    if ngrid_x-2>nthreads:
        chunksize=(ngrid_x-2)//nthreads
    else:
        chunksize=1
    arr1[:]=arr[:]
    
    with nogil, parallel(num_threads=nthreads):
        for i in prange(1,ngrid_x-1,schedule='static',chunksize=chunksize):
            for j in range(1,ngrid_y-1):
                for k in range(1,ngrid_z-1):
                    
                    arr1[i,j,k]+=arr[i-1, j, k]
                    arr1[i,j,k]+=arr[i, j-1, k]
                    arr1[i,j,k]+=arr[i, j, k-1]
                    arr1[i,j,k]+=arr[i-1, j-1, k]
                    arr1[i,j,k]+=arr[i-1, j, k-1]
                    arr1[i,j,k]+=arr[i, j-1, k-1]
                    arr1[i,j,k]+=arr[i-1, j-1, k-1]
                    
                    arr1[i,j,k]+=arr[i+1, j, k]
                    arr1[i,j,k]+=arr[i, j+1, k]
                    arr1[i,j,k]+=arr[i, j, k+1]
                    arr1[i,j,k]+=arr[i+1, j+1, k]
                    arr1[i,j,k]+=arr[i+1, j, k+1]
                    arr1[i,j,k]+=arr[i, j+1, k+1]
                    arr1[i,j,k]+=arr[i+1, j+1, k+1]
                    
                    arr1[i,j,k]+=arr[i-1, j+1, k]
                    arr1[i,j,k]+=arr[i+1, j-1, k]
                    
                    arr1[i,j,k]+=arr[i+1, j, k-1]
                    arr1[i,j,k]+=arr[i-1, j, k+1]
                    
                    arr1[i,j,k]+=arr[i, j+1, k-1]
                    arr1[i,j,k]+=arr[i, j-1, k+1]
                    
                    arr1[i,j,k]+=arr[i+1, j-1, k-1]
                    arr1[i,j,k]+=arr[i-1, j+1, k-1]
                    arr1[i,j,k]+=arr[i-1, j-1, k+1]
                    arr1[i,j,k]+=arr[i-1, j+1, k+1]
                    arr1[i,j,k]+=arr[i+1, j-1, k+1]
                    arr1[i,j,k]+=arr[i+1, j+1, k-1]
                    
                    arr1[i,j,k]/=27.0
                    
