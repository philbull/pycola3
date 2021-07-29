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

cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.embedsignature(True)
def grad_phi(            np.ndarray[np.float32_t, ndim=3] sx,
                         np.ndarray[np.float32_t, ndim=3] sy,
                         np.ndarray[np.float32_t, ndim=3] sz,
                         
                         np.ndarray[np.float32_t, ndim=3] sx2,
                         np.ndarray[np.float32_t, ndim=3] sy2,
                         np.ndarray[np.float32_t, ndim=3] sz2,
                         
                         np.ndarray[np.float32_t, ndim=3] velx,
                         np.ndarray[np.float32_t, ndim=3] vely,
                         np.ndarray[np.float32_t, ndim=3] velz,
                         
                         np.int32_t npart_x,
                         np.int32_t npart_y,
                         np.int32_t npart_z,
                         np.ndarray[np.float32_t, ndim=3] field, # field is phi
                         np.int32_t ngrid_x,
                         np.int32_t ngrid_y,
                         np.int32_t ngrid_z,
                         
                         np.float32_t cellsize,
                         np.float32_t gridcellsize,
                         
                         np.float32_t growth,
                         np.float32_t growth2,
                         np.ndarray[np.float32_t, ndim=1] offset
                 ):
    r"""
    :math:`\vspace{-1mm}`

    Calculate the gradient of a potential by issuing a call to 
    :func:`acceleration.grad_phi_engine`. Arguments are the same as in 
    :func:`acceleration.grad_phi_engine` but internally it sets::
    
        add_lagrangian_position=1
        beta1=1
        beta2=0
    
    And ``vel``\ :sub:`i` is set to zero first, i.e. :math:`\bm{v}_{\mathrm{in}}=0`.

    """
    velx*=0.0
    vely*=0.0
    velz*=0.0
    
    grad_phi_engine(     sx,#dummy
                         sy,#dummy
                         sz,#dummy
                         
                         velx,
                         vely,
                         velz,
                         
                         
                         sx,
                         sy,
                         sz,
                         
                         sx2,
                         sy2,
                         sz2,
                         
                         1.0,
                         0.0,
                         
                         npart_x,
                         npart_y,
                         npart_z,
                         field, 
                         ngrid_x,
                         ngrid_y,
                         ngrid_z,
                         cellsize,
                         gridcellsize,
                         growth,
                         growth2,
                         offset,
                         1)
                    






@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.embedsignature(True)
def grad_phi_engine(     np.ndarray[np.float32_t, ndim=3] posx,
                         np.ndarray[np.float32_t, ndim=3] posy,
                         np.ndarray[np.float32_t, ndim=3] posz,
                         
                         np.ndarray[np.float32_t, ndim=3] velx,
                         np.ndarray[np.float32_t, ndim=3] vely,
                         np.ndarray[np.float32_t, ndim=3] velz,
                         
                         
                         np.ndarray[np.float32_t, ndim=3] sx,
                         np.ndarray[np.float32_t, ndim=3] sy,
                         np.ndarray[np.float32_t, ndim=3] sz,
                         
                         np.ndarray[np.float32_t, ndim=3] sx2,
                         np.ndarray[np.float32_t, ndim=3] sy2,
                         np.ndarray[np.float32_t, ndim=3] sz2,
                         
                         np.float32_t beta1,
                         np.float32_t beta2,
                         
                         np.int32_t npart_x,
                         np.int32_t npart_y,
                         np.int32_t npart_z,
                         np.ndarray[np.float32_t, ndim=3] field, # field is phi
                         np.int32_t ngrid_x,
                         np.int32_t ngrid_y,
                         np.int32_t ngrid_z,
                         np.float32_t cellsize,
                         np.float32_t gridcellsize,
                         np.float32_t growth,
                         np.float32_t growth2,
                         np.ndarray[np.float32_t, ndim=1] offset,
                         np.int32_t add_lagrangian_position):
    r""" 
    :math:`\vspace{-1mm}`

    Calculate particle accelerations using a finite difference scheme 
    to save memory. In particular, the function evaluates the following 
    equation for each particle:
    
    .. math::
      :nowrap:

      \begin{eqnarray}
          \bm{v}_{\mathrm{out}} = \bm{v}_{\mathrm{in}} + \beta_1 \bm{\nabla}\phi+\beta_2\bigg(g_1\bm{s}^{(1)}+g_1\bm{s}^{(2)}\bigg)
      \end{eqnarray}
      
    
    If ``add_lagrangian_position=0``, then :math:`\bm{\nabla}\phi` is 
    evaluated at position ``pos``:sub:`i` for each particle. If 
    ``add_lagrangian_position=1``, the gradient is evaluated at the 
    particle position given by its Lagrangian position plus a 
    displacement :math:`g_1\bm{s}^{(1)}+g_1\bm{s}^{(2)}+`\ ``offset``. 
    In the latter case, periodic boundary conditions are assumed.
    
    **Arguments**:
    
    * ``posx,posy,posz`` -- 3-dim float32 arrays. Not used when 
      ``add_lagrangian_position=1``. See above.
    
    * ``velx,vely,velz`` -- 3-dim float32 arrays, containing the components 
      of :math:`\bm{v}_{\mathrm{in}}` above. Overwritten on output to 
      contain :math:`\bm{v}_{\mathrm{out}}`.
    
    * ``sx,sy,sz`` -- 3-dim float32 arrays, containing the components 
      of :math:`\bm{s}^{(1)}` above.
    
    * ``sx2,sy2,sz2`` -- 3-dim float32 arrays, containing the components 
      of :math:`\bm{s}^{(2)}` above.
    
    * ``beta1,beta2`` -- float32. Equal :math:`\beta_1` and 
      :math:`\beta_2` above, respectively.
    
    * ``npart_x,npart_y,npart_z`` -- int32, giving the size of the 
      input particle arrays (e.g. ``sx``).
    
    * ``field`` -- 3-dim float32 array, containing the potential 
      :math:`\phi` above.
    
    * ``ngrid_x,ngrid_y,ngrid_z`` -- int32, giving the size of the 
      ``field`` array.
    
    * ``cellsize`` -- float32. The interparticle spacing in physical units.
    
    * ``gridcellsize`` -- float32. The grid spacing in physical units.
    
    * ``growth, growth2`` -- float32. Equal :math:`g_1` and 
      :math:g_2` above, respectively.
    
    * ``offset`` -- 1-dim float32 array. Not used when 
      ``add_lagrangian_position=0``. See above.
    
    * ``add_lagrangian_position`` -- int32. See above.
    
    **Result**:
    
    * The arrays ``velx,vely,velz`` are updated according to 
      the equation above.
    
    **Algorithm**:
    
    Use a 4-point finite difference scheme  [#f2]_ combined with a 
    bi-linear interpolation in the orthogonal directions. The 
    coefficients for the 4-pt finite difference are derived below with 
    `SymPy <http://sympy.org/en/index.html>`_. Use this piece of code to 
    generate coefficients for higher/lower-order difference schemes as 
    needed::
    
        >>> from sympy import *
        >>> a,da,dx,x=var('a da dx x')
        >>> fa,fp1,fm1,fp2=var('fa fp1 fm1 fp2')
        >>> d1,d2,d3=var('d1 d2 d3')
        >>> f=Function('f')
        >>> deriv_dict={f(a):fa,Subs(diff(f(x),x),(x,),(a,)) : d1, 
        ...                     Subs(diff(f(x),x,x),(x,),(a,)):d2,
        ...                     Subs(diff(f(x),x,x,x),(x,),(a,)):d3}
        >>> ftaylorTemp=series(f(x),x,a,4)#Call this only once if using SymPy<=0.7.5. 
        >>>                               #Repeated calls are buggy.
        >>>                               #Problem fixed on github.
        >>> ftaylor=(ftaylorTemp.xreplace(deriv_dict)).removeO() 
        >>> #ftaylor=ftaylor.subs({x:x-a}) # Uncomment this if using SymPy<=0.7.5.
        >>>                                # Fixed on github.
        >>> sol=solve([ftaylor.subs({x:a+da})-fp1, ftaylor.subs({x:a+2*da})-fp2, 
        ...            ftaylor.subs({x:a-da})-fm1],[d1,d2,d3],solution_dict=True)
        >>> res = diff(ftaylor,x).subs({x:a+dx*da}).subs(deriv_dict).subs(sol)
        >>> (6*da*res).expand().collect(fm1).collect(fp2).collect(fp1).collect(fa)
        f_0*(9*dx**2 - 12*dx - 3) + fm1*(-3*dx**2 + 6*dx - 2) +
        fp1*(-9*dx**2 + 6*dx + 6) + fp2*(3*dx**2 - 1)

    .. rubric:: Footnotes

    .. [#f2] Not to be confused with the 4-pt calculation of the second-order initial conditions.
    
    """
    
    cdef int i1, j1, k1, i,j,k
    cdef np.float32_t xpos, ypos, zpos,dL
    cdef np.float32_t edge_x,edge_y,edge_z
    cdef np.float32_t dx, dy, dz, dx2, dy2, dz2
    cdef np.float32_t c00x,c1px,c2px,c1mx
    cdef np.float32_t c00y,c1py,c2py,c1my
    cdef np.float32_t c00z,c1pz,c2pz,c1mz
    cdef int i1p,j1p,k1p,  i2p,j2p,k2p,  i1m,j1m,k1m
    cdef np.float32_t ax,ay,az
    
    
    
    edge_x = (<np.float32_t> ngrid_x) - 0.0001
    edge_y = (<np.float32_t> ngrid_y) - 0.0001
    edge_z = (<np.float32_t> ngrid_z) - 0.0001

    dL=gridcellsize


    from cython.parallel cimport prange,parallel
    cdef int nthreads
    from multiprocessing import cpu_count
    nthreads=cpu_count()
    #print 'nthreads = ',  nthreads
    

    with nogil, parallel(num_threads=nthreads):
        for i in prange(npart_x,schedule='static',chunksize=npart_x//nthreads): 
            for j in range(npart_y):
                for k in range(npart_z):
    #for i in range():
                    
                    ##
                    if (add_lagrangian_position):
                        if (growth2<1.e-10):
                            xpos = sx[i,j,k] * growth  + (<np.float32_t> (i) )*cellsize + (<np.float32_t> (ngrid_x) )*gridcellsize
                            ypos = sy[i,j,k] * growth  + (<np.float32_t> (j) )*cellsize + (<np.float32_t> (ngrid_y) )*gridcellsize
                            zpos = sz[i,j,k] * growth  + (<np.float32_t> (k) )*cellsize + (<np.float32_t> (ngrid_z) )*gridcellsize
                        else:
                            xpos = sx[i,j,k] * growth + sx2[i,j,k] * growth2  + (<np.float32_t> (i) )*cellsize + (<np.float32_t> (ngrid_x) )*gridcellsize
                            ypos = sy[i,j,k] * growth + sy2[i,j,k] * growth2  + (<np.float32_t> (j) )*cellsize + (<np.float32_t> (ngrid_y) )*gridcellsize
                            zpos = sz[i,j,k] * growth + sz2[i,j,k] * growth2  + (<np.float32_t> (k) )*cellsize + (<np.float32_t> (ngrid_z) )*gridcellsize
                    
                    
                        xpos = xpos + offset[0]
                        ypos = ypos + offset[1]
                        zpos = zpos + offset[2]
                    
                        xpos = xpos % ((<np.float32_t> (ngrid_x) )*gridcellsize)
                        ypos = ypos % ((<np.float32_t> (ngrid_y) )*gridcellsize)
                        zpos = zpos % ((<np.float32_t> (ngrid_z) )*gridcellsize)
                    else:
                        xpos = posx[i,j,k]
                        ypos = posy[i,j,k]
                        zpos = posz[i,j,k]
                    ##
                    xpos = xpos/gridcellsize
                    ypos = ypos/gridcellsize
                    zpos = zpos/gridcellsize
                    
                    if xpos<0.0001: 
                        xpos=0.0001
                    if xpos>edge_x:
                        xpos=edge_x
                    
                    if ypos<0.0001: 
                        ypos=0.0001
                    if ypos>edge_y:
                        ypos=edge_y
                        
                    if zpos<0.0001: 
                        zpos=0.0001
                    if zpos>edge_z:
                        zpos=edge_z
                        
                    
                    i1  = <int> (xpos)
                    j1  = <int> (ypos)
                    k1  = <int> (zpos)
                    
                    dx = xpos - (<float> i1)
                    dy = ypos - (<float> j1)
                    dz = zpos - (<float> k1)
                    
                    dx2 =  1.0 - dx
                    dy2 =  1.0 - dy
                    dz2 =  1.0 - dz

                    i1p = i1+1
                    j1p = j1+1
                    k1p = k1+1
                    
                    if i1p >= ngrid_x: i1p = 0
                    if j1p >= ngrid_y: j1p = 0
                    if k1p >= ngrid_z: k1p = 0
                    
                    i1m = i1-1
                    j1m = j1-1
                    k1m = k1-1
                    
                    if i1m < 0: i1m = ngrid_x-1
                    if j1m < 0: j1m = ngrid_y-1
                    if k1m < 0: k1m = ngrid_z-1
                    
                    
                    i2p = i1p+1
                    j2p = j1p+1
                    k2p = k1p+1
                    
                    if i2p >= ngrid_x: i2p = 0
                    if j2p >= ngrid_y: j2p = 0
                    if k2p >= ngrid_z: k2p = 0
                    
                    
                    
                    c00x = 3.0 * (-1.0 + dx * (-4.0 + 3.0 * dx))/(6.0*dL)
                    c1px = ( 6.0 + 3.0 * dx * ( 2.0 - 3.0 * dx))/(6.0*dL)
                    c1mx = (-2.0 - 3.0 * dx * (-2.0 +       dx))/(6.0*dL)
                    c2px = (-1.0 + 3.0 * dx *               dx) /(6.0*dL)
                    
                    ax       = (
                                    field[i2p ,j1p ,k1p ] * c2px  * dy  * dz  +
                                    field[i1p ,j1p ,k1p ] * c1px  * dy  * dz  +
                                    field[i1  ,j1p ,k1p ] * c00x  * dy  * dz  +
                                    field[i1m ,j1p ,k1p ] * c1mx  * dy  * dz  +
                                                                              
                                    field[i2p ,j1  ,k1p ] * c2px  * dy2 * dz  +
                                    field[i1p ,j1  ,k1p ] * c1px  * dy2 * dz  +
                                    field[i1  ,j1  ,k1p ] * c00x  * dy2 * dz  +
                                    field[i1m ,j1  ,k1p ] * c1mx  * dy2 * dz  +
                                    
                                    field[i2p ,j1p ,k1  ] * c2px  * dy  * dz2 +
                                    field[i1p ,j1p ,k1  ] * c1px  * dy  * dz2 +
                                    field[i1  ,j1p ,k1  ] * c00x  * dy  * dz2 +
                                    field[i1m ,j1p ,k1  ] * c1mx  * dy  * dz2 +
                                                                  
                                                                  
                                    field[i2p ,j1  ,k1  ] * c2px  * dy2 * dz2 +
                                    field[i1p ,j1  ,k1  ] * c1px  * dy2 * dz2 +
                                    field[i1  ,j1  ,k1  ] * c00x  * dy2 * dz2 +
                                    field[i1m ,j1  ,k1  ] * c1mx  * dy2 * dz2 
                                 )
                    
                    c00y = 3.0 * (-1.0 + dy * (-4.0 + 3.0 * dy))/(6.0*dL)
                    c1py = ( 6.0 + 3.0 * dy * ( 2.0 - 3.0 * dy))/(6.0*dL)
                    c1my = (-2.0 - 3.0 * dy * (-2.0 +       dy))/(6.0*dL)
                    c2py = (-1.0 + 3.0 * dy *               dy) /(6.0*dL)
                    
                    ay       = (
                                     field[i1p ,j2p ,k1p ] * c2py * dx  * dz  +
                                     field[i1p ,j1p ,k1p ] * c1py * dx  * dz  +
                                     field[i1p ,j1  ,k1p ] * c00y * dx  * dz  +
                                     field[i1p ,j1m ,k1p ] * c1my * dx  * dz  +
                                     
                                     field[i1  ,j2p ,k1p ] * c2py * dx2 * dz  +
                                     field[i1  ,j1p ,k1p ] * c1py * dx2 * dz  +
                                     field[i1  ,j1  ,k1p ] * c00y * dx2 * dz  +
                                     field[i1  ,j1m ,k1p ] * c1my * dx2 * dz  +
                                                                
                                     field[i1p ,j2p ,k1  ] * c2py * dx  * dz2 +
                                     field[i1p ,j1p ,k1  ] * c1py * dx  * dz2 +
                                     field[i1p ,j1  ,k1  ] * c00y * dx  * dz2 +
                                     field[i1p ,j1m ,k1  ] * c1my * dx  * dz2 +
                                                                
                                     field[i1  ,j2p ,k1  ] * c2py * dx2 * dz2 +
                                     field[i1  ,j1p ,k1  ] * c1py * dx2 * dz2 +
                                     field[i1  ,j1  ,k1  ] * c00y * dx2 * dz2 +
                                     field[i1  ,j1m ,k1  ] * c1my * dx2 * dz2 
                                 )
                    
                    
                    c00z = 3.0 * (-1.0 + dz * (-4.0 + 3.0 * dz))/(6.0*dL)
                    c1pz = ( 6.0 + 3.0 * dz * ( 2.0 - 3.0 * dz))/(6.0*dL)
                    c1mz = (-2.0 - 3.0 * dz * (-2.0 +       dz))/(6.0*dL)
                    c2pz = (-1.0 + 3.0 * dz *               dz) /(6.0*dL)
                    
                    
                    az       = (
                                     field[i1p ,j1p ,k2p ] * c2pz * dx  * dy  +
                                     field[i1p ,j1p ,k1p ] * c1pz * dx  * dy  +
                                     field[i1p ,j1p ,k1  ] * c00z * dx  * dy  +
                                     field[i1p ,j1p ,k1m ] * c1mz * dx  * dy  +
                                                                
                                     field[i1  ,j1p ,k2p ] * c2pz * dx2 * dy  +
                                     field[i1  ,j1p ,k1p ] * c1pz * dx2 * dy  +
                                     field[i1  ,j1p ,k1  ] * c00z * dx2 * dy  +
                                     field[i1  ,j1p ,k1m ] * c1mz * dx2 * dy  +
                                                                
                                     field[i1p ,j1  ,k2p ] * c2pz * dx  * dy2 +
                                     field[i1p ,j1  ,k1p ] * c1pz * dx  * dy2 +
                                     field[i1p ,j1  ,k1  ] * c00z * dx  * dy2 +
                                     field[i1p ,j1  ,k1m ] * c1mz * dx  * dy2 +
                                                                
                                     field[i1  ,j1  ,k2p ] * c2pz * dx2 * dy2 +
                                     field[i1  ,j1  ,k1p ] * c1pz * dx2 * dy2 +
                                     field[i1  ,j1  ,k1  ] * c00z * dx2 * dy2 +
                                     field[i1  ,j1  ,k1m ] * c1mz * dx2 * dy2 
                                 )
                                 
                    # update velocities with accelerations
                    if (growth2<1.e-10):
                        velx[i,j,k] = velx[i,j,k] + beta1 * ax + beta2 * (growth * sx[i,j,k] )
                        vely[i,j,k] = vely[i,j,k] + beta1 * ay + beta2 * (growth * sy[i,j,k] )
                        velz[i,j,k] = velz[i,j,k] + beta1 * az + beta2 * (growth * sz[i,j,k] )
                    else:
                        velx[i,j,k] = velx[i,j,k] + beta1 * ax + beta2 * (growth * sx[i,j,k] + growth2 * sx2[i,j,k])
                        vely[i,j,k] = vely[i,j,k] + beta1 * ay + beta2 * (growth * sy[i,j,k] + growth2 * sy2[i,j,k])
                        velz[i,j,k] = velz[i,j,k] + beta1 * az + beta2 * (growth * sz[i,j,k] + growth2 * sz2[i,j,k])
