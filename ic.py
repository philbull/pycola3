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


def _power_spectrum(filename):
    """
    :math:`\\vspace{-1mm}`

    
    Self-explanatory
    """
    from scipy import interpolate
    from numpy import loadtxt
    
    (k_arr,p_arr)=loadtxt(filename).transpose()
        
    return interpolate.interp1d(k_arr,p_arr,kind='linear')




def initial_positions(sx,sy,sz,sx2,sy2,sz2,cellsize,growth_factor,growth_factor_2lpt,
                            ngrid_x,
                            ngrid_y,
                            ngrid_z,
                            gridcellsize,offset=[0.0,0.0,0.0]
                          ):
    """ 
    :math:`\\vspace{-1mm}`

    Add the Lagrangian particle position to the 2LPT displacement to 
    obtain the Eulerian position. Periodic boundary conditions are assumed. 
    
    
    **Arguments**:

    * ``sx,sy,sz`` --  3-dim NumPy arrays containing the 
      components of the particle 
      displacements today as calculated in the ZA.
       
    * ``sx2,sy2,sz2`` --  3-dim NumPy arrays containing the 
      components of the second order particle 
      displacements today as calculated in 2LPT. 
    
    
    * ``cellsize`` --  a float. The inter particle spacing in Lagrangian space.
    
    * ``growth_factor`` --  a float. The linear growth factor for the 
      redshift for which the Eulerian positions are requested.

    * ``growth_factor_2lpt`` --  a float. The second order growth factor for the 
      redshift for which the Eulerian positions are requested.
    
    * ``ngrid_x``, ``ngrid_y``, ``ngrid_z`` -- integers. The grid size 
      of the box. Only used together with ``gridcellsize`` below to find the 
      physical size of the box, which is needed to apply the periodic 
      boundary conditions.
    
    * ``gridcellsize`` -- a float. The grid spacing of the box. 
    
    * ``offset`` -- a list of three floats (default: ``[0.0,0.0,0.0]``). 
      Offset the Eulerian particle positions by this amount. Useful for 
      placing refined subregions at their proper locations inside a 
      bigger box.
    
    **Return**:
    
    * ``(px,py,pz)`` --   a tuple, where ``p``:sub:`i`
      is a 3-dim single precision NumPy array containing the ``i``-th 
      component (``i`` = ``x``, ``y``, ``z``) of the particle 
      Eulerian position.
      
    **Example**:
    
    In this example we generate the initial conditions in 2LPT, and 
    then plot a slice through the 2LPT realization at redshift of zero.

        >>> from ic import ic_za,ic_2lpt,initial_positions
        >>> sx,sy,sz=ic_za('camb_matterpower_z0.dat',npart=128)
        Memory allocation done
        Plans created
        Power spectrum read.
        Randoms done.
        Nyquists fixed
        sx fft ready
        sy fft ready
        sz fft ready
        >>> sx2,sy2,sz2 = ic_2lpt( 100.0/float(sx.shape[0]),sx,sy,sz,
        ...                        growth_2pt_calc=0.1)
        >>> px,py,pz = initial_positions(sx,sy,sz,sx2,sy2,sz2,100./128.,1.0,1.0,
        ...            1, # only ngrid_i*gridcellsize=boxsize is relevant here
        ...            1,
        ...            1,
        ...            100.0)
        >>> import matplotlib.pyplot as plt # needs matplotlib to be installed
        >>> import numpy as np
        >>> ind=np.where(pz<3)
        >>> px_slice=px[ind]
        >>> py_slice=py[ind]
        >>> plt.figure(figsize=(10,10))
        <matplotlib.figure.Figure object at 0x7f21044d2e10>
        >>> plt.scatter(px_slice,py_slice,marker='.',alpha=0.03,color='r')
        <matplotlib.collections.PathCollection object at 0x7f2102cd3290>
        >>> plt.show()



    
    
    """ 
    
    from numpy import indices
    
    npart_x,npart_y,npart_z=sx.shape
    
    px,py,pz=indices((npart_x,npart_y,npart_z),dtype='float32')
    
    
    px *= cellsize
    py *= cellsize
    pz *= cellsize
    
    px += float(ngrid_x)*gridcellsize + offset[0]
    py += float(ngrid_y)*gridcellsize + offset[1]
    pz += float(ngrid_z)*gridcellsize + offset[2]
    
    px += sx * growth_factor
    py += sy * growth_factor
    pz += sz * growth_factor
    
    px += sx2 * growth_factor_2lpt
    py += sy2 * growth_factor_2lpt
    pz += sz2 * growth_factor_2lpt
    
    px %= float(ngrid_x)*gridcellsize
    py %= float(ngrid_y)*gridcellsize
    pz %= float(ngrid_z)*gridcellsize
    
    
    
    return px,py,pz



def import_music_snapshot(hdf5_filename,boxsize,level0='09',level1=None):
    """ 
    :math:`\\vspace{-1mm}`

    Import a MUSIC snapshot calculated in the ZA. 
    
    **Arguments**:

    * ``hdf5_filename`` --  a string. Gives the filename for the `HDF5 <http://www.hdfgroup.org/HDF5/>`_ 
      file, which MUSIC outputs.
       
    * ``boxsize`` -- a float. The size of the full simulation box in :math:`\mathrm{Mpc}/h`.
    
    * ``level0`` --   a two-character string (default: ``'09'``). A 
      MUSIC level covering the whole box. With the settings below, it should 
      equal ``levelmin`` from the MUSIC configuration file for the 
      finest such level.
    
    * ``level1`` --   a two-character string (default: ``None``). A fine 
      MUSIC level covering the refined subvolume. With the settings below, it 
      should equal ``levelmax`` from the MUSIC configuration file for 
      the finest such level.
    
    
    **Return**:
    
    * if ``level1`` is ``None``: ``(sx,sy,sz)`` --   a tuple, where ``s``:sub:`i`
      is a 3-dim single precision NumPy array containing the ``i``-th 
      component (``i`` = ``x``, ``y``, ``z``) of the particle 
      displacements today as calculated in the ZA. ``s``:sub:`i` are the 
      displacements for the ``level0`` particles.
    
    * if ``level1`` is not ``None``: 
      ``(sx,sy,sz,sx_zoom,sy_zoom,sz_zoom,offset)`` --   a tuple, where:
      
      - ``s``:sub:`i` and ``s``:sub:`i`\ ``_zoom`` are 3-dim single precision NumPy arrays containing 
        the ``i``-th component (``i`` = ``x``, ``y``, ``z``) of the 
        particle displacements today as calculated in the ZA. ``s``:sub:`i` are 
        the displacements for the crude level (``level0``) particles; 
        while ``s``:sub:`i`\ ``_zoom`` are the displacements for the fine level 
        (``level1``) particles in the refined subvolume. 
        
      - ``offset`` -- a list of three integers giving the crude-grid 
        index coordinates of the origin of the fine grid.
    
    
    .. note:: 
       pyCOLA requires specific values for some keywords in the MUSIC 
       configuration file. Those are::
    
           zstart = 0
           align_top = yes
           use_2LPT = no
           format = generic
    
       Also, if ``level1`` is not ``None``, pyCOLA assumes that one 
       uses only one (usually the finest) refinement level (``level1``) on the subvolume 
       of interest. Then the following needs to hold::
           
           levelmin<levelmax
           ref_extent!=1.0,1.0,1.0
           
       See the included :download:`ics.conf <./ics.conf>` for an example.
    
    

    """
    
    import h5py
    print "Starting import ..."
    ss = h5py.File(hdf5_filename, "r")
    
    # for some reason MUSIC pads with 4 elements the displacement arrays  
    # when `format = generic` in the MUSIC conf file.
    # In my checks, this did not depend on the settings for the
    # padding and overlap keywords. So, hardwiring this number...
    
    sx      = ss['level_0'+level0+'_DM_dx'].value[4:-4, 4:-4,  4:-4]*boxsize
    sy      = ss['level_0'+level0+'_DM_dy'].value[4:-4, 4:-4,  4:-4]*boxsize
    sz      = ss['level_0'+level0+'_DM_dz'].value[4:-4, 4:-4,  4:-4]*boxsize
    if not (level1 is None):
        offset=[ss['header']['grid_off_x'].value[-1],
                                 ss['header']['grid_off_y'].value[-1],
                                 ss['header']['grid_off_z'].value[-1]]
        sx_zoom = ss['level_0'+level1+'_DM_dx'].value[4:-4, 4:-4,  4:-4]*boxsize
        sy_zoom = ss['level_0'+level1+'_DM_dy'].value[4:-4, 4:-4,  4:-4]*boxsize
        sz_zoom = ss['level_0'+level1+'_DM_dz'].value[4:-4, 4:-4,  4:-4]*boxsize
        del ss
        print "... done"
        return sx, sy, sz, sx_zoom, sy_zoom, sz_zoom,offset
    else:
        del ss
        print "... done"
        return sx, sy, sz



def ic_2lpt( 
           cellsize,
           
           sx,
           sy,
           sz,
           sx_zoom = None,
           sy_zoom = None,
           sz_zoom = None,
           
           boxsize=100.00, 
           ngrid_x_lpt=128,ngrid_y_lpt=128,ngrid_z_lpt=128,

           
           cellsize_zoom=0,offset_zoom=None,BBox_in=None,
           growth_2pt_calc=0.05,
           with_4pt_rule = False,
           factor_4pt=2.0
           ):
    """ 
    :math:`\\vspace{-1mm}`

    
    Given a set of displacements calculated in the ZA at redshift 
    zero, find the corresponding second order displacement. Works 
    with a single grid of particles, as well as with one refined subvolume.
    
    **Arguments**:

    * ``cellsize`` --  a float. The inter-particle spacing in Lagrangian space.
    
    * ``sx,sy,sz`` --  3-dim NumPy arrays containing the 
      components of the particle displacements today as calculated in 
      the ZA. These particles should cover the whole box. If a refined 
      subvolume is provided, the crude particles which reside inside 
      that subvolume are discarded and replaced with the fine 
      particles.
       
    * ``sx_zoom,sy_zoom,sz_zoom`` --  3-dim NumPy arrays containing the 
      components of the particle 
      ZA displacements today for a refined subvolume (default: ``None``).
    
    * ``boxsize`` -- a float (default: ``100.0``). Gives the size of the 
      simulation box in Mpc/h.
    
    * ``ngrid_x_lpt,ngrid_y_lpt,ngrid_z_lpt`` -- integers 
      (default: ``128``). Provide the size of the PM grid, which the algorithm 
      uses to calculate the 2LPT displacements.
    
    * ``cellsize_zoom`` -- a float (default: ``0``). The inter-particle 
      spacing in Lagrangian space for the refined subvolume, if such is 
      provided. If not, ``cellsize_zoom`` must be set to zero 
      (default), as that is used as a check for the presence of that 
      subvolume.
    
    * ``offset_zoom`` -- a 3-vector of floats (default: ``None``). Gives the 
      physical coordinates of the origin of the refinement region 
      relative to the the origin of the full box.
    
    * ``BBox_in`` -- a 3x2 array of integers (default: ``None``). It has the 
      form ``[[i0,i1],[j0,j1],[k0,k1]]``, which gives the bounding box 
      for the refinement region in units of the crude particles 
      Lagrangian index. Thus, the particles with displacements 
      ``sx|sy|sz[i0:i1,j0:j1,k0:k1]`` are replaced with the fine 
      particles with displacements ``sx_zoom|sy_zoom|sz_zoom``.
    
    * ``growth_2pt_calc`` --  a float (default: ``0.05``). The 
      linear growth factor used internally in the 2LPT calculation. A 
      value of 0.05 gives excellent cross-correlation between the 2LPT 
      field returned by this function, and the 2LPT returned using the 
      usual fft tricks. Yet, some irrelevant short-scale noise is 
      present, which one may decide to filter out. That noise is most 
      probably due to lack of force accuracy for too low 
      ``growth_2pt_calc``. Experiment with this value as 
      needed.

    * ``with_4pt_rule`` -- a boolean (default: ``False``). See :func:`ic.ic_2lpt_engine`.
    
    * ``factor_4pt`` -- a float (default: ``2.0``). See :func:`ic.ic_2lpt_engine`.
    
    
    
    **Return**:
    
    * If no refined subregion is supplied (indicated by 
      ``cellsize_zoom=0``), then return:
      
      ``(sx2,sy2,sz2)`` --  3-dim NumPy 
      arrays containing the components of the second order particle 
      displacements today as calculated in 2LPT. 

    * If a refined subregion is supplied (indicated by 
      ``cellsize_zoom>0``), then return:
      
      ``(sx2,sy2,sz2,sx2_zoom,sy2_zoom,sz2_zoom)``
      
      The first three arrays are as 
      above. The last three give the second order displacements today 
      for the particles in the refined subvolume.
    
    **Example**:

    Generate a realization for the displacement field in the ZA; 
    calculate the corresponding second order displacement field; calculate 
    the rms displacements; then show a projection of one of the 
    components.

        >>> from ic import ic_za,ic_2lpt
        >>> sx,sy,sz=ic_za('camb_matterpower_z0.dat',npart=128)
        Memory allocation done
        Plans created
        Power spectrum read.
        Randoms done.
        Nyquists fixed
        sx fft ready
        sy fft ready
        sz fft ready
        >>> sx2,sy2,sz2 = ic_2lpt( 100.0/float(sx.shape[0]),sx,sy,sz,
        ...                        growth_2pt_calc=0.1)
        >>> ((sx**2+sy**2+sz**2).mean())**0.5/0.7    # ~10
        11.605451188108798
        >>> ((sx2**2+sy2**2+sz2**2).mean())**0.5/0.7 # ~2
        2.3447627779313525
        >>> import matplotlib.pyplot as plt # needs matplotlib to be installed!
        >>> plt.imshow(sx.mean(axis=2))
        <matplotlib.image.AxesImage object at 0x7fc4603697d0>
        >>> plt.show()
        >>> plt.imshow(sy2.mean(axis=2))
        <matplotlib.image.AxesImage object at 0x7fc4603697d0>
        >>> plt.show()
    
    **Algorithm**: 
        
    This function issues a call to :func:`ic.ic_2lpt_engine`. See the 
    Algorithm section of that function for details.
    
    """
    


    from ic import ic_2lpt_engine
    res = ic_2lpt_engine(  
             sx,                                                    
             sy,                                                    
             sz,                                                    
             cellsize,                                                             
                                                
             ngrid_x_lpt,ngrid_y_lpt,ngrid_z_lpt,                        
             boxsize/float(ngrid_x_lpt), # assumes cube
             
                                                                                   
             with_2lpt=False,                                                       
             sx2_full = None,                                       
             sy2_full = None,                                       
             sz2_full = None,                                       
                                                                                   
             cellsize_zoom = cellsize_zoom,             
             BBox_in = BBox_in,                       
             sx_full_zoom =  sx_zoom,              
             sy_full_zoom =  sy_zoom,                                                 
             sz_full_zoom =  sz_zoom,      
             sx2_full_zoom = None,            
             sy2_full_zoom = None,            
             sz2_full_zoom = None,            
                                              
             
             offset_zoom=offset_zoom,
             growth_2pt_calc=growth_2pt_calc
             )
    if (cellsize_zoom!=0):
        sx_,sy_,sz_,sx2,sy2,sz2,sx_zoom_,sy_zoom_,sz_zoom_,sx2_zoom,sy2_zoom,sz2_zoom =res
    else:
        sx_,sy_,sz_,sx2,sy2,sz2 =res
    
    del sx_,sy_,sz_ #These have higher order corrections unlike the original *_full arrays, which are 'exact'. So, discard.
    if (cellsize_zoom!=0):
            del sx_zoom_,sy_zoom_,sz_zoom_
            return sx2,sy2,sz2, sx2_zoom,sy2_zoom,sz2_zoom
    return  sx2,sy2,sz2
















def ic_2lpt_engine(                                       
         sx_full,
         sy_full,
         sz_full,
         cellsize,
         
         
         ngrid_x,ngrid_y,ngrid_z,
         gridcellsize,
         
         growth_2pt_calc=0.05,
         
         with_4pt_rule = False,
         factor_4pt=2.0,
         
         with_2lpt=False,
         sx2_full = None,
         sy2_full = None,
         sz2_full = None,
         
         cellsize_zoom = 0,
         BBox_in = None,
         sx_full_zoom =  None,
         sy_full_zoom =  None,
         sz_full_zoom =  None,
         sx2_full_zoom = None,
         sy2_full_zoom = None,
         sz2_full_zoom = None,
         
         
         offset_zoom=None):
    r""" 
    :math:`\vspace{-1mm}`

    The same as :func:`ic.ic_2lpt` above, but calculates the 2LPT displacements for the particles in the 
    COLA volume as generated by same particles displaced 
    according to the 2LPT of the full box. (todo: *expand this!*) In fact, :func:`ic.ic_2lpt` works 
    by making a call to this function.
    
    **Arguments**:
    
    * ``sx_full,sy_full,sz_full`` --  3-dim NumPy arrays containing the 
      components of the particle displacements today as calculated in 
      the ZA of the full box. These particles should cover the whole box. If a refined 
      subvolume is provided, the crude particles which reside inside 
      that subvolume are discarded and replaced with the fine 
      particles.
       
    * ``cellsize`` --  a float. The inter-particle spacing in Lagrangian space.

    * ``ngrid_x,ngrid_y,ngrid_z`` -- integers. Provide the size of the 
      PM grid, which the algorithm 
      uses to calculate the 2LPT displacements.
    
    * ``gridcellsize`` --float. Provide the grid spacing of the PM 
      grid, which the algorithm 
      uses to calculate the 2LPT displacements.
    
    * ``growth_2pt_calc`` --  a float (default: ``0.05``). The 
      linear growth factor used internally in the 2LPT calculation. A 
      value of 0.05 gives excellent cross-correlation between the 2LPT 
      field returned by this function, and the 2LPT returned using the 
      usual fft tricks for a 100:math:`\mathrm{Mpc}/h` box. Yet, some 
      irrelevant short-scale noise is present, which one may decide to 
      filter out. That noise is probably due to lack of force accuracy 
      for too low ``growth_2pt_calc``. Experiment with this value as 
      needed.
      
      
    * ``with_4pt_rule`` -- a boolean (default: ``False``). Whether to use 
      the 4-point force rule to evaluate the ZA and 2LPT displacements 
      in the COLA region. See the Algorithm section below. If set to 
      False, it uses the 2-point force rule.
    
    * ``factor_4pt`` -- a float, different from ``1.0`` (default: 
      ``2.0``). Used for the 4-point 
      force rule. See the Algorithm section below.
      
    * ``with_2lpt`` -- a boolean (default: ``False``). Whether the second 
      order displacement field  over the full box is provided. One must 
      provide it if the COLA volume is different from the full box. 
      Only if they are the same (as in the case of ``ic_2lpt()``) can 
      one set ``with_2lpt=False``.
       
    * ``sx2_full,sy2_full,sz2_full`` -- 3-dim NumPy float arrays giving the second 
      order displacement field  over the full box. Needs ``with_2lpt=True``.       
       
    * The rest of the input is as in :func:`ic.ic_2lpt`, with all LPT 
      quantities provided for the whole box.
       
    
    **Return**:
    
    * If no refined subregion is supplied (indicated by 
      ``cellsize_zoom=0``), then return:

      ``(sx,sy,sz,sx2,sy2,sz2)`` --  3-dim NumPy 
      arrays containing the components of the first and second (``s``:sub:`i`\ ``2``) 
      order particle displacements today as calculated in 2LPT in the 
      COLA volume. 

    * If a refined subregion is supplied (indicated by 
      ``cellsize_zoom>0``), then return:
      
    ``(sx,sy,sz,sx2,sy2,sz2,sx_zoom,sy_zoom,sz_zoom,sx2_zoom,sy2_zoom,sz2_zoom)``
      
      The first 6 arrays are as 
      above. The last 6 give the second order displacements today 
      for the particles in the refined subvolume of the COLA volume.
    
    **Algorithm**:
    
    The first-order and second-order displacements, 
    :math:`\bm{s}^{(1)}_{\mathrm{COLA}}` and 
    :math:`\bm{s}^{(2)}_{\mathrm{COLA}}`, in the COLA volume at 
    redshift zero are calculated according to the following 2-pt or 
    4-pt (denoted by subscript) equations:
    
    .. math::
      :nowrap:

      \begin{eqnarray}
          \bm{s}_{\mathrm{COLA},\mathrm{2pt}}^{(1)}    & = & - \frac{1}{2g}                      \left[\bm{F}(g,\beta g^2)-\bm{F}(-g,\beta g^2)\right] \\
          \bm{s}_{\mathrm{COLA},\mathrm{2pt}}^{(2)}    & = & - \frac{\alpha}{2g^2}                 \left[\bm{F}(g,\beta g^2)+\bm{F}(-g,\beta g^2)\right] \\
          \bm{s}_{\mathrm{COLA},\mathrm{4pt}}^{(1)}    & = & - \frac{1}{2g}     \frac{a^2}{a^2-1}\bigg[\bm{F}(g,\beta g^2)-\bm{F}(-g,\beta g^2)-\\
                                                     &   & \quad \quad \quad \quad \quad \quad - \frac{1}{a^3}\bigg(\bm{F}\left(a g,\beta a^2 g^2\right)-\bm{F}\left(-a g,\beta a^2 g^2\right)\bigg)\bigg] \\
          \bm{s}_{\mathrm{COLA},\mathrm{4pt}}^{(2)}    & = & - \frac{\alpha}{2g^2}\frac{a^2}{a^2-1}\bigg[\bm{F}(g,\beta g^2)+\bm{F}(-g,\beta g^2)-\\
                                                     &   & \quad \quad \quad \quad \quad \quad - \frac{1}{a^4}\bigg(\bm{F}\left(a g,\beta a^2 g^2\right)+\bm{F}\left(-a g,\beta a^2 g^2\right)\bigg)\bigg] 
      \end{eqnarray}
    
    where:
    
      :math:`a=` ``factor_4pt``
      
      :math:`g=` ``growth_2pt_calc``
      
      if ``with_2lpt`` then:
          
          :math:`\beta=1` and :math:`\alpha=(3/10)\Omega_{m}^{1/143}`
      
      else:
        
          :math:`\beta=0` and :math:`\alpha=(3/7)\Omega_{m}^{1/143}`
    
    The factors of :math:`\Omega_{m}^{1/143}` (:math:`\Omega_m` being 
    the matter density today) are needed to rescale the second order 
    displacements to matter domination and are correct to 
    :math:`\mathcal{O}(\max(10^{-4},g^3/143))` in 
    :math:`\Lambda\mathrm{CDM}`. The force :math:`\bm{F}(g_1,g_2)` is 
    given by:
    
    .. math::
      :nowrap:

      \begin{eqnarray}
          \bm{F}(g_1,g_2) = \bm{\nabla}\nabla^{-2}\delta\left[g_1\bm{s}_{\mathrm{full}}^{(1)}+g_2\Omega_{m}^{-1/143}\bm{s}_{\mathrm{full}}^{(2)}\right]
      \end{eqnarray}
    
    where :math:`\delta[\bm{s}]` is the cloud-in-cell fractional 
    overdensity calculated from a grid of particles displaced by the 
    input displacement vector  field :math:`\bm{s}`. Above, 
    :math:`\bm{s}_{\mathrm{full}}^{(1)}/\bm{s}_{\mathrm{full}}^{(2)}` are 
    the input first/second-order input displacement fields calculated 
    in the full box at redshift zero. 
    
    It is important to note that implicitly for each particle at 
    Lagrangian position :math:`\bm{q}`, the force 
    :math:`\bm{F}(g_1,g_2)` is evaluated at the corresponding Eulerian position:
    :math:`\bm{q}+g_1\bm{s}_{\mathrm{full}}^{(1)}+g_2\Omega_{m}^{-1/143}\bm{s}_{\mathrm{full}}^{(2)}`.
    
    As noted above, ``with_2lpt=False`` is only allowed if the COLA 
    volume covers the full box volume. In that case, 
    :math:`\bm{s}_{\mathrm{full}}^{(2)}` is not needed as input since 
    :math:`\beta=0`. Instead, the output 
    :math:`\bm{s}_{\mathrm{COLA}}^{(2)}` can be used as a good 
    approximation to :math:`\bm{s}_{\mathrm{full}}^{(2)}`. This fact 
    is used in :func:`ic.ic_2lpt` to calculate 
    :math:`\bm{s}_{\mathrm{full}}^{(2)}` from 
    :math:`\bm{s}_{\mathrm{full}}^{(1)}`.
      
    
    .. note:: If ``with_4pt_rule=False``, then the first/second order 
       displacements receive corrections at third/fourth order. If 
       ``with_4pt_rule=True``, then those corrections are fifth/sixth 
       order. However, when using the 4-point rule instead of the 
       2-point rule, one must make two more force evaluations at a 
       slightly different growth factor given by 
       ``growth_2pt_calc*factor_4pt``. Since the code is single 
       precision and is using a simple PM grid to evaluate forces, one 
       cannot make ``factor_4pt`` and ``growth_2pt_calc`` too small due 
       to noise issues. Thus, when comparing the 2-pt and 4-pt rule, we 
       should assume ``factor_4pt>1``. And again due to numerical 
       precision issues, one cannot choose ``factor_4pt`` to be too 
       close to one; hence, the default value of ``2.0``. 
       
       Therefore, as the higher order corrections for the 4-pt rule are 
       proportional to powers of ``growth_2pt_calc*factor_4pt``, one 
       may be better off using the 2-pt rule (the default) in this 
       particular implementation. Yet for codes where force accuracy is 
       not an issue, one may consider using the 4-pt rule. Thus, its 
       inclusion in this code is mostly done as an illustration.
       
    """
             
    
    
    
    
    from numpy import float64,float32
    
    if (cellsize_zoom!=0):
        cellsize_zoom=float32(cellsize_zoom)
        offset_zoom=offset_zoom.astype('float32')
    
    
    npart_x, npart_y, npart_z = sx_full.shape
    if (cellsize_zoom!=0):
        npart_x_zoom, npart_y_zoom, npart_z_zoom = sx_full_zoom.shape
  

    from numpy import zeros,array
    
    
    sx =  zeros((npart_x,npart_y,npart_z),dtype='float32')
    sy =  zeros((npart_x,npart_y,npart_z),dtype='float32')
    sz =  zeros((npart_x,npart_y,npart_z),dtype='float32')
    
    sx_minus =  zeros((npart_x,npart_y,npart_z),dtype='float32')
    sy_minus =  zeros((npart_x,npart_y,npart_z),dtype='float32')
    sz_minus =  zeros((npart_x,npart_y,npart_z),dtype='float32')

    if (cellsize_zoom!=0):
        sx_zoom =  zeros((npart_x_zoom,npart_y_zoom,npart_z_zoom),dtype='float32')
        sy_zoom =  zeros((npart_x_zoom,npart_y_zoom,npart_z_zoom),dtype='float32')
        sz_zoom =  zeros((npart_x_zoom,npart_y_zoom,npart_z_zoom),dtype='float32')
    
        sx_minus_zoom =  zeros((npart_x_zoom,npart_y_zoom,npart_z_zoom),dtype='float32')
        sy_minus_zoom =  zeros((npart_x_zoom,npart_y_zoom,npart_z_zoom),dtype='float32')
        sz_minus_zoom =  zeros((npart_x_zoom,npart_y_zoom,npart_z_zoom),dtype='float32')

    if (with_4pt_rule):
        sx_4pt_minus =  zeros((npart_x,npart_y,npart_z),dtype='float32')
        sy_4pt_minus =  zeros((npart_x,npart_y,npart_z),dtype='float32')
        sz_4pt_minus =  zeros((npart_x,npart_y,npart_z),dtype='float32')
        
        sx_4pt =  zeros((npart_x,npart_y,npart_z),dtype='float32')
        sy_4pt =  zeros((npart_x,npart_y,npart_z),dtype='float32')
        sz_4pt =  zeros((npart_x,npart_y,npart_z),dtype='float32')
    
        if (cellsize_zoom!=0): 
            sx_4pt_minus_zoom =  zeros((npart_x_zoom,npart_y_zoom,npart_z_zoom),dtype='float32')
            sy_4pt_minus_zoom =  zeros((npart_x_zoom,npart_y_zoom,npart_z_zoom),dtype='float32')
            sz_4pt_minus_zoom =  zeros((npart_x_zoom,npart_y_zoom,npart_z_zoom),dtype='float32')
        
            sx_4pt_zoom =  zeros((npart_x_zoom,npart_y_zoom,npart_z_zoom),dtype='float32')
            sy_4pt_zoom =  zeros((npart_x_zoom,npart_y_zoom,npart_z_zoom),dtype='float32')
            sz_4pt_zoom =  zeros((npart_x_zoom,npart_y_zoom,npart_z_zoom),dtype='float32')
    else:
        sx_4pt_minus =  0.0 
        sy_4pt_minus =  0.0
        sz_4pt_minus =  0.0
        
        sx_4pt =  0.0
        sy_4pt =  0.0
        sz_4pt =  0.0
    
        if (cellsize_zoom!=0): 
            sx_4pt_minus_zoom =  0.0
            sy_4pt_minus_zoom =  0.0
            sz_4pt_minus_zoom =  0.0
        
            sx_4pt_zoom =  0.0
            sy_4pt_zoom =  0.0
            sz_4pt_zoom =  0.0
    
    
    from potential import  get_phi, initialize_density
    from cic import CICDeposit_3
    from acceleration import grad_phi
    
    

###
    density,den_k,den_fft,phi_fft  = initialize_density(ngrid_x,ngrid_y,ngrid_z)
    density.fill(0.0)   
    Om=0.274
    dd= Om**(1./143.) # this is a good enough approximation at early times and is ~0.95
    if with_2lpt:
       cc=3./10.*dd
       L2=growth_2pt_calc*growth_2pt_calc/dd
    else:
       cc=3./7.*dd
       L2=0.0
    
    gridcellsize=float32(gridcellsize)
    growth_2pt_calc=float32(growth_2pt_calc)
    L2=float32(L2)
    
    offset=array([0.0,0.0,0.0],dtype='float32')
    if (cellsize_zoom==0):
        BBox_in=array([[0,0],[0,0],[0,0]],dtype='int32')
    if not with_2lpt:
        sx2_full = zeros((0,0,0),dtype='float32')
        sy2_full = zeros((0,0,0),dtype='float32')
        sz2_full = zeros((0,0,0),dtype='float32')
        
    CICDeposit_3(         sx_full,
                          sy_full,
                          sz_full,
                          sx2_full,
                          sy2_full,
                          sz2_full,
                          density,
                          
                          cellsize,gridcellsize,
                          1,
                          growth_2pt_calc,
                          L2,
                          
                          BBox_in,
                          offset,1)
                          
    if (cellsize_zoom!=0):
        CICDeposit_3(sx_full_zoom,
                          sy_full_zoom,
                          sz_full_zoom,
                          sx2_full_zoom,
                          sy2_full_zoom,
                          sz2_full_zoom,
                          density,
                          
                          cellsize_zoom,gridcellsize,
                          1,
                          growth_2pt_calc,
                          L2,

                          array([[0,0],[0,0],[0,0]],dtype='int32'),
                          offset_zoom,1)


    density -= 1.0
    
    #print "den ic",density.mean(dtype=float64)
    if (with_4pt_rule):
        density *= -0.5/growth_2pt_calc/(1.0-1.0/factor_4pt/factor_4pt)
    else:
        density *= -0.5/growth_2pt_calc
    
    get_phi(density, den_k, den_fft, phi_fft, ngrid_x,ngrid_y,ngrid_z, gridcellsize)
    phi = density # density now holds phi, so rename it
    grad_phi( sx_full, sy_full, sz_full,sx2_full, sy2_full, sz2_full, sx, sy, sz, npart_x,npart_y,npart_z, phi,
                ngrid_x,ngrid_y,ngrid_z,cellsize,gridcellsize, 
                growth_2pt_calc,
                L2,offset)
                
    if (cellsize_zoom!=0):
        grad_phi( sx_full_zoom, sy_full_zoom, sz_full_zoom,sx2_full_zoom, sy2_full_zoom, sz2_full_zoom, sx_zoom, sy_zoom, sz_zoom, npart_x_zoom,npart_y_zoom,npart_z_zoom, phi,
                ngrid_x,ngrid_y,ngrid_z,cellsize_zoom,gridcellsize, 
                growth_2pt_calc,
                L2,offset_zoom)

                         
#######
    density.fill(0.0)   

                          

    CICDeposit_3(sx_full,
                          sy_full,
                          sz_full,
                          sx2_full,
                          sy2_full,
                          sz2_full,
                          density,
                          
                          cellsize,gridcellsize,
                          1,
                         -growth_2pt_calc,
                          L2,
                          BBox_in,
                          offset,1)

    if (cellsize_zoom!=0):
        CICDeposit_3(sx_full_zoom,                                             
                          sy_full_zoom,                                             
                          sz_full_zoom,                                             
                          sx2_full_zoom,                                            
                          sy2_full_zoom,                                            
                          sz2_full_zoom,                                            
                          density,                                               
                          
                          cellsize_zoom,gridcellsize,                                                                                
                          1,                                                     
                         -growth_2pt_calc,                                    
                          L2,        
                          
                          array([[0,0],[0,0],[0,0]],dtype='int32'),
                          offset_zoom,1)
                                                                                 
                                                                                 
    density -= 1.0  
    
    #print "den ic",density.mean(dtype=float64)
    if (with_4pt_rule):
        density *= -0.5/growth_2pt_calc/(1.0-1.0/factor_4pt/factor_4pt)                                        
    else:
        density *= -0.5/growth_2pt_calc

    
    get_phi(density, den_k, den_fft, phi_fft, ngrid_x,ngrid_y,ngrid_z, gridcellsize)
    phi = density # density now holds phi, so rename it

    grad_phi( sx_full, sy_full, sz_full,sx2_full, sy2_full, sz2_full, sx_minus, sy_minus, sz_minus, npart_x,npart_y,npart_z, phi,
                ngrid_x,ngrid_y,ngrid_z,cellsize,gridcellsize, 
               -growth_2pt_calc,
                L2,offset)
    if (cellsize_zoom!=0):
        grad_phi( sx_full_zoom, sy_full_zoom, sz_full_zoom,sx2_full_zoom, sy2_full_zoom, sz2_full_zoom, sx_minus_zoom, sy_minus_zoom, sz_minus_zoom, npart_x_zoom,npart_y_zoom,npart_z_zoom, phi,
                ngrid_x,ngrid_y,ngrid_z,cellsize_zoom,gridcellsize, 
               -growth_2pt_calc,
                L2,offset_zoom)

###### 
###### 
###### Two more force evaluations in the case of a 4pt rule.
###### 
###### 

    if (with_4pt_rule):
        density.fill(0.0)   
        CICDeposit_3(sx_full,
                          sy_full,
                          sz_full,
                          sx2_full,
                          sy2_full,
                          sz2_full,
                          density,
                          
                          cellsize,gridcellsize,
                          1,
                          growth_2pt_calc*factor_4pt,
                          L2*factor_4pt*factor_4pt,
                          
                          BBox_in,
                          offset,1)
        if (cellsize_zoom!=0):
            CICDeposit_3(sx_full_zoom,
                          sy_full_zoom,
                          sz_full_zoom,
                          sx2_full_zoom,
                          sy2_full_zoom,
                          sz2_full_zoom,
                          density,
                          
                          cellsize_zoom,gridcellsize,
                          1,
                          growth_2pt_calc*factor_4pt,
                          L2*factor_4pt*factor_4pt,

                          array([[0,0],[0,0],[0,0]],dtype='int32'),
                          offset_zoom,1)


        density -= 1.0
    
    #print "den ic",density.mean(dtype=float64)
    
        density *= -0.5/growth_2pt_calc/(1.0-1.0/factor_4pt/factor_4pt)*(-1.0/factor_4pt**3)

    
        get_phi(density, den_k, den_fft, phi_fft, ngrid_x,ngrid_y,ngrid_z, gridcellsize)
        phi = density # density now holds phi, so rename it
        grad_phi( sx_full, sy_full, sz_full,sx2_full, sy2_full, sz2_full, sx_4pt, sy_4pt, sz_4pt, npart_x,npart_y,npart_z, phi,
                ngrid_x,ngrid_y,ngrid_z,cellsize,gridcellsize, 
                growth_2pt_calc*factor_4pt,
                L2*factor_4pt*factor_4pt,offset)
        if (cellsize_zoom!=0):
            grad_phi( sx_full_zoom, sy_full_zoom, sz_full_zoom,sx2_full_zoom, sy2_full_zoom, sz2_full_zoom, sx_4pt_zoom, sy_4pt_zoom, sz_4pt_zoom, npart_x_zoom,npart_y_zoom,npart_z_zoom, phi,
                ngrid_x,ngrid_y,ngrid_z,cellsize_zoom,gridcellsize, 
                growth_2pt_calc*factor_4pt,
                L2*factor_4pt*factor_4pt,offset_zoom)

                         
#######
        density.fill(0.0)   

                          

        CICDeposit_3(sx_full,
                          sy_full,
                          sz_full,
                          sx2_full,
                          sy2_full,
                          sz2_full,
                          density,
                          
                          cellsize,gridcellsize,
                          1,
                         -growth_2pt_calc*factor_4pt,
                          L2*factor_4pt*factor_4pt,
                          BBox_in,
                          offset,1)
        if (cellsize_zoom!=0):
            CICDeposit_3(sx_full_zoom,                                             
                          sy_full_zoom,                                             
                          sz_full_zoom,                                             
                          sx2_full_zoom,                                            
                          sy2_full_zoom,                                            
                          sz2_full_zoom,                                            
                          density,                                               
                          
                          cellsize_zoom,gridcellsize,                                                                                
                          1,                                                     
                         -growth_2pt_calc*factor_4pt,                                    
                          L2*factor_4pt*factor_4pt,        
                          
                          array([[0,0],[0,0],[0,0]],dtype='int32'),
                          offset_zoom,1)
                                                                                 
                                                                                 
        density -= 1.0  
                     
        density *= -0.5/growth_2pt_calc/(1.0-1.0/factor_4pt/factor_4pt)*(-1.0/factor_4pt**3)                                    


    
        get_phi(density, den_k, den_fft, phi_fft, ngrid_x,ngrid_y,ngrid_z, gridcellsize)
        phi = density # density now holds phi, so rename it

        grad_phi( sx_full, sy_full, sz_full,sx2_full, sy2_full, sz2_full, sx_4pt_minus, sy_4pt_minus, sz_4pt_minus, npart_x,npart_y,npart_z, phi,
                ngrid_x,ngrid_y,ngrid_z,cellsize,gridcellsize, 
               -growth_2pt_calc*factor_4pt,
                L2*factor_4pt*factor_4pt,offset)
        if (cellsize_zoom!=0):
            grad_phi( sx_full_zoom, sy_full_zoom, sz_full_zoom,sx2_full_zoom, sy2_full_zoom, sz2_full_zoom, sx_4pt_minus_zoom, sy_4pt_minus_zoom, sz_4pt_minus_zoom, npart_x_zoom,npart_y_zoom,npart_z_zoom, phi,
                ngrid_x,ngrid_y,ngrid_z,cellsize_zoom,gridcellsize, 
               -growth_2pt_calc*factor_4pt,
                L2*factor_4pt*factor_4pt,offset_zoom)

###### 
###### 
###### Done with the two more force evaluations in the case of a 4pt rule.
###### 
###### 

    del density,den_k, den_fft,phi,phi_fft
    
    
    if (cellsize_zoom!=0): # the variables *_4pt* (except factor_4pt) are init'd to zero if 4pt rule is not requested
        sx2_zoom = (sx_zoom+sx_minus_zoom+(sx_4pt_zoom+sx_4pt_minus_zoom)/factor_4pt)*cc/growth_2pt_calc
        sy2_zoom = (sy_zoom+sy_minus_zoom+(sy_4pt_zoom+sy_4pt_minus_zoom)/factor_4pt)*cc/growth_2pt_calc
        sz2_zoom = (sz_zoom+sz_minus_zoom+(sz_4pt_zoom+sz_4pt_minus_zoom)/factor_4pt)*cc/growth_2pt_calc
    
        sx_zoom+=sx_4pt_zoom - (sx_minus_zoom + sx_4pt_minus_zoom)
        sy_zoom+=sy_4pt_zoom - (sy_minus_zoom + sy_4pt_minus_zoom)
        sz_zoom+=sz_4pt_zoom - (sz_minus_zoom + sz_4pt_minus_zoom)
    
    sx2 = (sx+sx_minus+(sx_4pt+sx_4pt_minus)/factor_4pt)*cc/growth_2pt_calc 
    sy2 = (sy+sy_minus+(sy_4pt+sy_4pt_minus)/factor_4pt)*cc/growth_2pt_calc
    sz2 = (sz+sz_minus+(sz_4pt+sz_4pt_minus)/factor_4pt)*cc/growth_2pt_calc
    
    sx+=sx_4pt - (sx_minus + sx_4pt_minus) 
    sy+=sy_4pt - (sy_minus + sy_4pt_minus)
    sz+=sz_4pt - (sz_minus + sz_4pt_minus)
    
    del sx_minus,sy_minus,sz_minus
    if (cellsize_zoom!=0):
        del sx_minus_zoom,sy_minus_zoom,sz_minus_zoom
    if (with_4pt_rule):
        del sx_4pt_minus,  sy_4pt_minus,  sz_4pt_minus
        if (cellsize_zoom!=0):    
            del sx_4pt_minus_zoom,sy_4pt_minus_zoom,sz_4pt_minus_zoom
        del sx_4pt,  sy_4pt,  sz_4pt
        if (cellsize_zoom!=0):
            del sx_4pt_zoom,sy_4pt_zoom,sz_4pt_zoom
    
    # The lines below fix the box-size irrotational condition for periodic boxes
    
    #Mx=sx.mean(axis=0,dtype=float64)
    #My=sy.mean(axis=1,dtype=float64)
    #Mz=sz.mean(axis=2,dtype=float64)
    #for i in range(npart_x):
        #sx[i,:,:]-=Mx
    #for i in range(npart_y):
        #sy[:,i,:]-=My
    #for i in range(npart_z):
        #sz[:,:,i]-=Mz

    #del Mx,My,Mz
    #Mx=sx2.mean(axis=0,dtype=float64)
    #My=sy2.mean(axis=1,dtype=float64)
    #Mz=sz2.mean(axis=2,dtype=float64)
    #for i in range(npart_x):
        #sx2[i,:,:]-=Mx
    #for i in range(npart_y):
        #sy2[:,i,:]-=My
    #for i in range(npart_z):
        #sz2[:,:,i]-=Mz

    #del Mx,My,Mz
    #Mx=sx_zoom.mean(axis=0,dtype=float64)
    #My=sy_zoom.mean(axis=1,dtype=float64)
    #Mz=sz_zoom.mean(axis=2,dtype=float64)
    #for i in range(npart_x_zoom):
        #sx_zoom[i,:,:]-=Mx
    #for i in range(npart_y_zoom):
        #sy_zoom[:,i,:]-=My
    #for i in range(npart_z_zoom):
        #sz_zoom[:,:,i]-=Mz

    #del Mx,My,Mz
    #Mx=sx2_zoom.mean(axis=0,dtype=float64)
    #My=sy2_zoom.mean(axis=1,dtype=float64)
    #Mz=sz2_zoom.mean(axis=2,dtype=float64)
    #for i in range(npart_x_zoom):
        #sx2_zoom[i,:,:]-=Mx
    #for i in range(npart_y_zoom):
        #sy2_zoom[:,i,:]-=My
    #for i in range(npart_z_zoom):
        #sz2_zoom[:,:,i]-=Mz
    #del Mx,My,Mz
    
    
    if (cellsize_zoom!=0):
        return sx,sy,sz,sx2,sy2,sz2,sx_zoom,sy_zoom,sz_zoom,sx2_zoom,sy2_zoom,sz2_zoom
    else:
        return sx,sy,sz,sx2,sy2,sz2





def ic_za(file_pk,boxsize=100.0,npart=64,init_seed=1234):
    """ 
    :math:`\\vspace{-1mm}`

    Generates Gaussian initial conditions for cosmological 
    simulations in the Zel'dovich appoximation (ZA) -- the first order 
    in Lagrangian Perturbation Theory (LPT).
    
    **Arguments**:

    * ``file_pk`` --  a string. Gives the filename for the plain text 
      file containing the matter power spectrum at redshift zero from 
      `CAMB <http://www.camb.info/>`_. For an example, see the included 
      :download:`camb_matterpower_z0.dat <./camb_matterpower_z0.dat>`.
        
    * ``boxsize`` -- a float (default: ``100.0``). Gives the size of the 
      simulation box in Mpc/h.
   
    * ``npart`` --   an integer (default: ``64``). The total number of 
      particles is ``npart``:sup:`3`.
    
    * ``init_seed`` --   an integer (default: ``1234``). The seed for the 
      random number generator.
    
    **Return**:
    
    * ``(sx,sy,sz)`` --   a tuple, where ``s``:sub:`i` is a 3-dim single 
      precision NumPy array containing the ``i``-th component 
      (``i`` = ``x``, ``y``, ``z``) of the particle displacements today as calculated 
      in the ZA.
    
            
    **Example**:
    
    Generate a realization for the displacement field; calculate the 
    rms displacements; and show a projection of one of the components.
    
        >>> from ic import ic_za
        >>> sx,sy,sz=ic_za('camb_matterpower_z0.dat')
        Memory allocation done
        Plans created
        Power spectrum read.
        Randoms done.
        Nyquists fixed
        sx fft ready
        sy fft ready
        sz fft ready
        >>> ((sx**2+sy**2+sz**2).mean())**0.5/0.7 # O(10) for our universe
        10.346006222040094
        >>> import matplotlib.pyplot as plt # needs matplotlib to be installed!
        >>> plt.imshow(sx.mean(axis=2))
        <matplotlib.image.AxesImage object at 0x7fc4603697d0>
        >>> plt.show()
        


    **Algorithm**:
    
        Implemented in the usual fft way.

    .. warning:: This function has been tested but not at the level of 
       trusting it for doing research. Use at your own risk.

    """

    #import sys
    #sys.path.append(dir)
    #sys.path.append('/home/user/Builds/pyFFTW-master-20140621/pyfftw')
    from numpy import pi,exp,sqrt
    #from cmath import exp
    #from math import sqrt
    import pyfftw
    from multiprocessing import cpu_count
    
    
    delta=2.0*pi/boxsize
    nyq=npart//2
    
    nalign=pyfftw.simd_alignment
    
    npart_pad = 2*(npart//2 + 1)
    
    sx_pad = pyfftw.n_byte_align_empty((npart,npart,npart_pad),nalign,'float32')
    sy_pad = pyfftw.n_byte_align_empty((npart,npart,npart_pad),nalign,'float32')
    sz_pad = pyfftw.n_byte_align_empty((npart,npart,npart_pad),nalign,'float32')
    
    sx = sx_pad[:,:,:npart]
    sy = sy_pad[:,:,:npart]
    sz = sz_pad[:,:,:npart]
    
    sx_k = sx_pad.view('complex64')
    sy_k = sy_pad.view('complex64')
    sz_k = sz_pad.view('complex64')
    
    
    print "Memory allocation done"
    
    nthreads=cpu_count()
    
    sx_fft=pyfftw.FFTW(sx_k,sx, axes=(0,1,2),direction='FFTW_BACKWARD',threads=nthreads,flags=('FFTW_ESTIMATE','FFTW_DESTROY_INPUT'))
    sy_fft=pyfftw.FFTW(sy_k,sy, axes=(0,1,2),direction='FFTW_BACKWARD',threads=nthreads,flags=('FFTW_ESTIMATE','FFTW_DESTROY_INPUT'))
    sz_fft=pyfftw.FFTW(sz_k,sz, axes=(0,1,2),direction='FFTW_BACKWARD',threads=nthreads,flags=('FFTW_ESTIMATE','FFTW_DESTROY_INPUT'))
    
    print "Plans created"
    
    p_of_k=_power_spectrum(file_pk)
    
    print "Power spectrum read."
    
    from numpy import random as rnd
    from numpy import indices
    
    rnd.seed(int(init_seed))
    
    x,y,z=indices((nyq+1,nyq+1,nyq+1),dtype='float32')
    
    
    c=[(npart -i) % npart for i in range(nyq+1)]
    
    w2 = x*x+y*y+z*z
    
    w2[0,0,0]=1.0 # irrelevant but a zero crashes
    
    
    
    amp  = sqrt(p_of_k(sqrt(w2)*delta)/boxsize**3) 
    amp *= boxsize/(2.0*pi) # fix dimensions of 1/k
    
    
    phi  = exp(1j*2.0*pi*rnd.uniform(0.0, 1.0, (nyq+1,nyq+1,nyq+1)))
    phi *= rnd.normal(0.0,1.0,(nyq+1,nyq+1,nyq+1))
    
    
    
    sx_k[0:nyq+1, 0:nyq+1, 0:nyq+1]=x*phi/w2*amp
    sy_k[0:nyq+1, 0:nyq+1, 0:nyq+1]=y*phi/w2*amp
    sz_k[0:nyq+1, 0:nyq+1, 0:nyq+1]=z*phi/w2*amp
    
    del phi
    
    phi  = exp(1j*2.0*pi*rnd.uniform(0.0, 1.0, (nyq+1,nyq+1,nyq+1)))
    phi *= rnd.normal(0.0,1.0,(nyq+1,nyq+1,nyq+1))
    
    sx_k[c, 0:nyq+1, 0:nyq+1]= - x*phi/w2*amp
    sy_k[c, 0:nyq+1, 0:nyq+1]=   y*phi/w2*amp
    sz_k[c, 0:nyq+1, 0:nyq+1]=   z*phi/w2*amp
    
    del phi
    
    phi  = exp(1j*2.0*pi*rnd.uniform(0.0, 1.0, (nyq+1,nyq+1,nyq+1)))
    phi *= rnd.normal(0.0,1.0,(nyq+1,nyq+1,nyq+1))
    
    sx_k[0:nyq+1, c, 0:nyq+1]=   x*phi/w2*amp
    sy_k[0:nyq+1, c, 0:nyq+1]= - y*phi/w2*amp
    sz_k[0:nyq+1, c, 0:nyq+1]=   z*phi/w2*amp
    
    del phi
    
    phi  = exp(1j*2.0*pi*rnd.uniform(0.0, 1.0, (nyq+1,nyq+1,nyq+1)))
    phi *= rnd.normal(0.0,1.0,(nyq+1,nyq+1,nyq+1))
    
    tmp = - x*phi/w2*amp
    sx_k[npart-1:nyq-1:-1, 0               , :]= tmp[1:nyq+1, 0      , :]
    sx_k[0               , npart-1:nyq-1:-1, :]= tmp[0      , 1:nyq+1, :]
    sx_k[0               , 0               , :]= tmp[0      , 0      , :]
    
    del tmp

    tmp = - y*phi/w2*amp
    sy_k[npart-1:nyq-1:-1, 0               , :]= tmp[1:nyq+1, 0      , :]
    sy_k[0               , npart-1:nyq-1:-1, :]= tmp[0      , 1:nyq+1, :]
    sy_k[0               , 0               , :]= tmp[0      , 0      , :]
    
    del tmp
    
    tmp =  z*phi/w2*amp
    sz_k[npart-1:nyq-1:-1, 0               , :]= tmp[1:nyq+1, 0      , :]
    sz_k[0               , npart-1:nyq-1:-1, :]= tmp[0      , 1:nyq+1, :]
    sz_k[0               , 0               , :]= tmp[0      , 0      , :]
    
    del phi,w2,amp,tmp
    
    

    print "Randoms done."

    sx_k[0,0,0]=0
    sy_k[0,0,0]=0
    sz_k[0,0,0]=0
    
    
    
    sx_k[npart-1:0:-1, npart-1:nyq-1:-1, [0,nyq]]  =  (sx_k[1:npart, 1:nyq+1,  [0,nyq]]).conjugate()
    sx_k[0           , npart-1:nyq-1:-1, [0,nyq]]  =  (sx_k[0      , 1:nyq+1,  [0,nyq]]).conjugate()
    sx_k[npart-1:0:-1, 0               , [0,nyq]]  =  (sx_k[1:npart, 0      ,  [0,nyq]]).conjugate()
                                                      
    sy_k[npart-1:0:-1, npart-1:nyq-1:-1, [0,nyq]]  =  (sy_k[1:npart, 1:nyq+1,  [0,nyq]]).conjugate()
    sy_k[0           , npart-1:nyq-1:-1, [0,nyq]]  =  (sy_k[0      , 1:nyq+1,  [0,nyq]]).conjugate()
    sy_k[npart-1:0:-1, 0               , [0,nyq]]  =  (sy_k[1:npart, 0      ,  [0,nyq]]).conjugate()
                                                      
    sz_k[npart-1:0:-1, npart-1:nyq-1:-1, [0,nyq]]  =  (sz_k[1:npart, 1:nyq+1,  [0,nyq]]).conjugate()
    sz_k[0           , npart-1:nyq-1:-1, [0,nyq]]  =  (sz_k[0      , 1:nyq+1,  [0,nyq]]).conjugate()
    sz_k[npart-1:0:-1, 0               , [0,nyq]]  =  (sz_k[1:npart, 0      ,  [0,nyq]]).conjugate()
    

    
    i = (x % nyq)+(y % nyq)+(z % nyq)==0
    del x,y,z
    
    
    sx_k[i]=sx_k[i].real
    sy_k[i]=sy_k[i].real
    sz_k[i]=sz_k[i].real
    
    del i
    
    print "Nyquists fixed"
    
    sx_fft(normalise_idft=False)
    #sX=sx.copy()
    del sx_k,sx_fft, sx_pad
    print "sx fft ready"
    
    sy_fft(normalise_idft=False)
    #sY=sy.copy()
    del sy_k,sy_fft, sy_pad
    print "sy fft ready"
    
    sz_fft(normalise_idft=False)
    #sZ=sz.copy()
    del sz_k,sz_fft, sz_pad
    print "sz fft ready"
    
    
    
    return sx,sy,sz

