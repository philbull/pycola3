#-----------------------------------------------------------------------------
#
#  The code in this file is loosely based on code from yt 
#  (http://yt-project.org/), so I kept the name of the function they used. 
#  Maybe the most important modification I did is
#  that now it is multithreaded. My modifications to their 
#  permissive-licensed file are GPL'd below. Please, keep that in 
#  mind.
#
#  -Svetlin
#
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
#
#  This file incorporates work covered by the following copyright and  
#  permission notice:  
#   
#        ===============================
#         The yt project licensing terms
#        ===============================
#        
#        yt is licensed under the terms of the Modified BSD License (also known as New
#        or Revised BSD), as follows:
#        
#        Copyright (c) 2013-, yt Development Team
#        Copyright (c) 2006-2013, Matthew Turk <matthewturk@gmail.com>
#        
#        All rights reserved.
#        
#        Redistribution and use in source and binary forms, with or without
#        modification, are permitted provided that the following conditions are met:
#        
#        Redistributions of source code must retain the above copyright notice, this
#        list of conditions and the following disclaimer.
#        
#        Redistributions in binary form must reproduce the above copyright notice, this
#        list of conditions and the following disclaimer in the documentation and/or
#        other materials provided with the distribution.
#        
#        Neither the name of the yt Development Team nor the names of its
#        contributors may be used to endorse or promote products derived from this
#        software without specific prior written permission.
#        
#        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#        ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#        WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#        DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
#        FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#        DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#        CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#        OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#        
#        About the yt Development Team
#        -----------------------------
#        
#        Matthew Turk began yt in 2006 and remains the project lead.  Over time yt has
#        grown to include contributions from a large number of individuals from many
#        diverse institutions, scientific, and technical backgrounds.
#        
#        Until the fall of 2013, yt was licensed under the GPLv3.  However, with consent
#        from all developers and on a public mailing list, yt has been relicensed under
#        the BSD 3-clause under a shared copyright model.  For more information, see:
#        http://lists.spacepope.org/pipermail/yt-dev-spacepope.org/2013-July/003239.html
#        All versions of yt prior to this licensing change are available under the
#        GPLv3; all subsequent versions are available under the BSD 3-clause license.
#        
#        The yt Development Team is the set of all contributors to the yt project.  This
#        includes all of the yt subprojects.
#        
#        The core team that coordinates development on BitBucket can be found here:
#        http://bitbucket.org/yt_analysis/ 
#        
#        
#        Our Copyright Policy
#        --------------------
#        
#        yt uses a shared copyright model. Each contributor maintains copyright
#        over their contributions to yt. But, it is important to note that these
#        contributions are typically only changes to the repositories. Thus, the yt
#        source code, in its entirety is not the copyright of any single person or
#        institution.  Instead, it is the collective copyright of the entire yt
#        Development Team.  If individual contributors want to maintain a record of what
#        changes/contributions they have specific copyright on, they should indicate
#        their copyright in the commit message of the change, when they commit the
#        change to one of the yt repositories.
#        
#        With this in mind, the following banner should be used in any source code file
#        to indicate the copyright and license terms:
#        
#        #-----------------------------------------------------------------------------
#        # Copyright (c) 2013, yt Development Team.
#        #
#        # Distributed under the terms of the Modified BSD License.
#        #
#        # The full license is in the file COPYING.txt, distributed with this software.
#        #-----------------------------------------------------------------------------






###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------


import numpy as np
cimport numpy as np
cimport cython



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.embedsignature(True)
def CICDeposit_3(
                 np.ndarray[np.float32_t, ndim=3] sx,
                 np.ndarray[np.float32_t, ndim=3] sy,
                 np.ndarray[np.float32_t, ndim=3] sz,
                 np.ndarray[np.float32_t, ndim=3] sx2,
                 np.ndarray[np.float32_t, ndim=3] sy2,
                 np.ndarray[np.float32_t, ndim=3] sz2,
                 np.ndarray[np.float32_t, ndim=3] field,
                 
                 np.float32_t cellsize,
                 np.float32_t gridcellsize,
                 
                 np.int32_t add_lagrangian_position,
                 np.float32_t growth_factor,     
                 np.float32_t growth_factor2,
                 
                 np.ndarray[np.int32_t, ndim=2] BBox,
                 
                 np.ndarray[np.float32_t, ndim=1] offset,
                 np.int32_t periodic):
    """
    :math:`\\vspace{-1mm}`

    Do a cloud-in-cell (CiC) assignment.
    
    **Arguments**:
    
    * ``sx,sy,sz`` -- 3-dim float32 arrays. If 
      ``add_lagrangian_position=0``, these arrays contain the three 
      components of the particle Eulerian positions. If 
      ``add_lagrangian_position=1``, they contain the particle linear 
      displacemens today.
      
    * ``sx2,sy2,sz2`` -- 3-dim float32 arrays. Contain the components of the particle 
      second-order displacemens today. Not used if 
      ``add_lagrangian_position=0``.
    
    * ``field`` -- 3-dim float32 array. Contains the density field after 
      the CiC assignment. 
    
    * ``cellsize``  -- float32. The interparticle spacing. 
    
    * ``gridcellsize``  -- float32. The grid spacing.
    
    * ``add_lagrangian_position`` -- int32. See the arguments above.
    
    * ``growth_factor`` -- float32. The linear growth factor by which to 
      multiply the ``s``:sub:`i` arrays when ``add_lagrangian_position=1``. Not   
      used when ``add_lagrangian_position=0``.
    
    * ``growth_factor2`` -- float32. The second-order growth factor by which to 
      multiply the ``s``:sub:`i`\ ``2`` arrays when ``add_lagrangian_position=1``. Not   
      used when ``add_lagrangian_position=0``.
    
    * ``BBox`` -- 2-dim int32 array. The bounding box of a Lagrangian 
      region which must be omitted in the CiC assignment. If we denote 
      its elements by ``[[i0,i1],[j0,j1],[k0,k1]]``, then the particles 
      with positions/displacements ``sx|sy|sz[i0:i1,j0:j1,k0:k1]`` are 
      skipped in the CiC. Used to carve a box inside a grid of crude 
      particles, to be filled later with fine particles.
      
    
    * ``offset`` -- 1-dim float32 array. The 3-vector offset of the 
      particles, in units of :math:`\mathrm{Mpc}/h`. Not   
      used when ``add_lagrangian_position=0``.
    
    * ``periodic`` -- int32. If ``periodic=1``, use periodic boundary 
      conditions.
    
    **Result**:
    
    * ``field`` is modified by adding to it the CiC assigned particles.
    
    .. warning:: The CiC is done on ``field`` without first 
       setting it to a default value. If you need to, fill it with zeros 
       before calling this function.   
    
    """
    
    cdef int npart_x,npart_y,npart_z

    cdef int i1, j1, k1, i,j,k
    cdef np.float32_t xpos, ypos, zpos
    cdef np.float32_t edge_x,edge_y,edge_z, mass
    
    cdef np.float32_t dx, dy, dz, dx2, dy2, dz2

    cdef int i1p,j1p,k1p

    npart_x = sx.shape[0]
    npart_y = sx.shape[1]
    npart_z = sx.shape[2]
    
    ngrid_x = field.shape[0]
    ngrid_y = field.shape[1]
    ngrid_z = field.shape[2]
    
    

    edge_x =  (<np.float32_t> ngrid_x)-0.0001 # Do not reduce this epsilon further, or risk
    edge_y =  (<np.float32_t> ngrid_y)-0.0001 # segfaults since using single precision.
    edge_z =  (<np.float32_t> ngrid_z)-0.0001 #


    mass  = (cellsize)**3/(gridcellsize)**3



    from cython.parallel cimport prange,parallel
    cdef int nthreads
    from multiprocessing import cpu_count
    nthreads=cpu_count()
    #print 'nthreads,npart_x = ',  nthreads,npart_x
    
    if npart_x>nthreads:
        chunksize=npart_x//nthreads
    else:
        chunksize=1
    
    
    with nogil, parallel(num_threads=nthreads):
        for i in prange(npart_x,schedule='static',chunksize=chunksize):
    #for i in range(npart_x):
            for j in range(npart_y):
                for k in range(npart_z):

                    if not( (BBox[0,0]<=i) and (BBox[0,1]>i) and 
                            (BBox[1,0]<=j) and (BBox[1,1]>j) and 
                            (BBox[2,0]<=k) and (BBox[2,1]>k) ): # exclude region in bbox
                        xpos = sx[i,j,k]
                        ypos = sy[i,j,k]
                        zpos = sz[i,j,k]

                        if (add_lagrangian_position==1):
                            if (growth_factor2<1.e-10):
                                xpos = xpos * growth_factor +  (<np.float32_t> i) * cellsize  
                                ypos = ypos * growth_factor +  (<np.float32_t> j) * cellsize  
                                zpos = zpos * growth_factor +  (<np.float32_t> k) * cellsize  
                            else:
                                xpos = xpos * growth_factor + sx2[i,j,k] * growth_factor2  + (<np.float32_t> i) * cellsize  
                                ypos = ypos * growth_factor + sy2[i,j,k] * growth_factor2  + (<np.float32_t> j) * cellsize  
                                zpos = zpos * growth_factor + sz2[i,j,k] * growth_factor2  + (<np.float32_t> k) * cellsize  
                            
                            xpos = xpos + offset[0]
                            ypos = ypos + offset[1]
                            zpos = zpos + offset[2]
                        
                        
                        
                        if (periodic==1):

                            xpos = xpos + (<np.float32_t> ngrid_x) * gridcellsize 
                            ypos = ypos + (<np.float32_t> ngrid_y) * gridcellsize 
                            zpos = zpos + (<np.float32_t> ngrid_z) * gridcellsize 
                            
                            xpos = xpos % ((<np.float32_t> ngrid_x) *gridcellsize )
                            ypos = ypos % ((<np.float32_t> ngrid_y) *gridcellsize )
                            zpos = zpos % ((<np.float32_t> ngrid_z) *gridcellsize )
                            
                        if (periodic==1) or ((xpos > 1.e-7) and (xpos < (((<np.float32_t> ngrid_x) *gridcellsize )-1.e-7)) and 
                            (ypos > 1.e-7) and (ypos < (((<np.float32_t> ngrid_y) *gridcellsize )-1.e-7)) and 
                            (zpos > 1.e-7) and (zpos < (((<np.float32_t> ngrid_z) *gridcellsize )-1.e-7))):
                            xpos = fclip(xpos/gridcellsize  ,0.0001, edge_x)
                            ypos = fclip(ypos/gridcellsize  ,0.0001, edge_y)
                            zpos = fclip(zpos/gridcellsize  ,0.0001, edge_z)
                            
                            i1  = <int> (xpos)
                            j1  = <int> (ypos)
                            k1  = <int> (zpos)
                    
                            i1p = i1+1
                            j1p = j1+1
                            k1p = k1+1
                            
                            if i1p >= ngrid_x: i1p=0
                            if j1p >= ngrid_y: j1p=0
                            if k1p >= ngrid_z: k1p=0
                    
                    
                            # Compute the weights
                            dx = xpos - (<np.float32_t> i1) 
                            dy = ypos - (<np.float32_t> j1) 
                            dz = zpos - (<np.float32_t> k1) 
                            dx2 =  1.0 - dx
                            dy2 =  1.0 - dy
                            dz2 =  1.0 - dz
                            

                            
                            # Interpolate from field into sumfield
                            field[i1p ,j1p ,k1p ] += mass * dx  * dy  * dz    
                            field[i1  ,j1p ,k1p ] += mass * dx2 * dy  * dz    
                            field[i1p ,j1  ,k1p ] += mass * dx  * dy2 * dz    
                            field[i1  ,j1  ,k1p ] += mass * dx2 * dy2 * dz    
                            field[i1p ,j1p ,k1  ] += mass * dx  * dy  * dz2   
                            field[i1  ,j1p ,k1  ] += mass * dx2 * dy  * dz2   
                            field[i1p ,j1  ,k1  ] += mass * dx  * dy2 * dz2   
                            field[i1  ,j1  ,k1  ] += mass * dx2 * dy2 * dz2   





"""
Shareable definitions for common fp/int Cython utilities



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

#cimport numpy as np
#cimport cython

cdef inline int imax(int i0, int i1) nogil:
    if i0 > i1: return i0
    return i1

cdef inline np.float32_t fmax(np.float32_t f0, np.float32_t f1) nogil:
    if f0 > f1: return f0
    return f1

cdef inline int imin(int i0, int i1) nogil:
    if i0 < i1: return i0
    return i1

cdef inline np.float32_t fmin(np.float32_t f0, np.float32_t f1) nogil:
    if f0 < f1: return f0
    return f1

cdef inline np.float32_t fabs(np.float32_t f0) nogil:
    if f0 < 0.0: return -f0
    return f0

cdef inline int iclip(int i, int a, int b) nogil:
    if i < a: return a
    if i > b: return b
    return i

cdef inline int i64clip(np.int64_t i, np.int64_t a, np.int64_t b) nogil:
    if i < a: return a
    if i > b: return b
    return i

cdef inline np.float32_t fclip(np.float32_t f,
                      np.float32_t a, np.float32_t b) nogil:
    return fmin(fmax(f, a), b)

