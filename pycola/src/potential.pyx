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
def  get_phi(np.ndarray[np.float32_t, ndim=3]  denphi,
         np.ndarray[np.complex64_t, ndim=3]  den_k,
         den_fft, phi_ifft,
         np.int32_t  ngrid_x,
         np.int32_t  ngrid_y,
         np.int32_t  ngrid_z,
         np.float32_t gridcellsize
         ):
    """
    :math:`\\vspace{-1mm}`
    
    Calculate the potential sourced by a given density field. Periodic 
    boundary conditions are assumed.
    
    **Arguments**: 
    
    * ``denphi,den_k,den_fft,phi_ifft`` -- these arrays 
      and classes are the output from a single call to 
      :func:`potential.initialize_density`::
      
            denphi,den_k,den_fft,phi_ifft = initialize_density(ngrid_x,ngrid_y,ngrid_z)
      
      The array ``denphi`` should then be assigned the values of the density 
      field, and then fed as an input to this function. It is 
      overwritten with the values of the potential on exit.
    
    * ``ngrid_x, ngrid_y, ngrid_z`` -- int32. The size of ``denphi``.
    
    * ``gridcellsize`` -- float32. Grid spacing of the PM grid in 
      physical units.
    
    **Result**:
    
    * ``denphi`` contains the potential on exit.
    
    **Algorithm**:
    
    Convolve the input density with the :math:`-1/k^2` kernel.

    """

    
    cdef int i,j,x,y,z,nyq_x,nyq_y,nyq_z
    cdef np.float32_t k2,delta2_x,delta2_y,delta2_z
    
    
    delta2_x=(2.0*np.pi/(gridcellsize*float(ngrid_x)))**2
    delta2_y=(2.0*np.pi/(gridcellsize*float(ngrid_y)))**2
    delta2_z=(2.0*np.pi/(gridcellsize*float(ngrid_z)))**2
    
    nyq_x=ngrid_x//2
    nyq_y=ngrid_y//2
    nyq_z=ngrid_z//2

    den_fft() # fft the density
    del den_fft
    
    
    from multiprocessing import cpu_count
    from cython.parallel cimport prange,parallel
    cdef int nthreads
    
    nthreads=cpu_count()
    #print 'nthreads = ',  nthreads
    
    chunk=ngrid_x//nthreads
    if chunk==0:
        chunk=1

    with nogil, parallel(num_threads=nthreads):
        for i in prange(ngrid_x,schedule='static',chunksize=chunk):    
    #for i in range(ngrid):
            x=i
            if x>nyq_x:
                x=ngrid_x-i
            for j in range(ngrid_y):
                y=j
                if y>nyq_y:
                    y=ngrid_y-j
                for z in range(nyq_z+1):
                    k2 =delta2_x * (<np.float32_t>(x*x)) + delta2_y * (<np.float32_t>(y*y))+delta2_z * (<np.float32_t>(z*z))
                    den_k[i,j,z] = - den_k[i,j,z] / k2 # for gaussian smoothing, change the -1/k2 kernel here to exp(-k2*smoothing_scale**2/2.0).
    
    den_k[0,0,0]=0
    
    phi_ifft(normalise_idft=True)
    del den_k, phi_ifft

@cython.embedsignature(True)
def initialize_density(ngrid_x,ngrid_y,ngrid_z):
    """
    :math:`\\vspace{-1mm}`
    
    Initialize the PM grid and its forward and inverse in-place Fourier 
    transforms. We use pyFFTW, which issues calls to the `fftw 
    library <http://www.fftw.org/>`_ to create the plans for the FFT.
    
    **Arguments**: 
    
    * ``ngrid_x,ngrid_y,ngrid_z`` -- integers, giving the size of the 
      PM grid.
      
    **Return**:
    
    * ``density`` -- a properly aligned 3-dim float32 array.
    
    * ``density_k`` -- a `view 
      <http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.view.html>`_ 
      of ``density`` as a 3-dim complex64 array. After executing the 
      forward fft plan, ``density_k`` contains the in-place fft'd 
      ``density``.
    
    * ``den_fft`` -- instance of the `FFTW class 
      <https://hgomersall.github.io/pyFFTW/pyfftw/pyfftw.html#pyfftw.FFTW>`_ 
      for computing the forward fft, which fft's ``density`` to give 
      ``density_k``. Creating the instance is equivalent to creating a 
      `fftw plan 
      <http://www.fftw.org/fftw3_doc/Using-Plans.html#Using-Plans>`_. 
      Calling the instance, executes the plan.
    
    * ``den_ifft`` -- instance of the FFTW class for computing the 
      inverse fft, which ifft's ``density_k`` to give back ``density`` 
      (up to normalization).
    
    
    """
    import pyfftw
    
    nalign=pyfftw.simd_alignment
    
    from multiprocessing import cpu_count
    cdef int nthreads
    nthreads=cpu_count()
    #print 'nthreads = ',  nthreads

    ngrid_pad = 2*(ngrid_z//2 + 1)
    
    density_pad = pyfftw.n_byte_align_empty((ngrid_x,ngrid_y,ngrid_pad),nalign,'float32')
    density = density_pad[:,:,:ngrid_z]
    density_k = density_pad.view('complex64')

    if nthreads>ngrid_z*2:
        nthreads=ngrid_z*2

    den_fft=pyfftw.FFTW(density,density_k, axes=(0,1,2),direction='FFTW_FORWARD',threads=nthreads,flags=('FFTW_ESTIMATE','FFTW_DESTROY_INPUT'))
    den_ifft=pyfftw.FFTW(density_k,density, axes=(0,1,2),direction='FFTW_BACKWARD',threads=nthreads,flags=('FFTW_ESTIMATE','FFTW_DESTROY_INPUT'))
    
    del density_pad
    
    return density,density_k,den_fft,den_ifft



    
