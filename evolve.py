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


def evolve( 
           cellsize,
           sx_full, sy_full, sz_full, 
           sx2_full, sy2_full, sz2_full, 
           FULL=False,
           
           cellsize_zoom=0,
           sx_full_zoom=None, 
           sy_full_zoom=None, 
           sz_full_zoom=None,
           sx2_full_zoom=None, 
           sy2_full_zoom=None, 
           sz2_full_zoom=None,
           offset_zoom=None,
           BBox_in=None,
           
           
           
           ngrid_x=None,ngrid_y=None,ngrid_z=None,
           gridcellsize=None,
           ngrid_x_lpt=None,ngrid_y_lpt=None,ngrid_z_lpt=None,
           gridcellsize_lpt=None,

           Om=0.274,
           Ol=1.0-0.274,
           
           a_initial=1./15., 
           a_final=1.0,
           n_steps=15,
           nCola=-2.5,
           
           save_to_file=False, 
           file_npz_out='tmp.npz'):
    r"""
    :math:`\vspace{-1mm}`

    Evolve a set of initial conditions forward in time using the COLA 
    method in both the spatial and temporal domains.

    **Arguments**:
    
    * ``cellsize`` --  a float. The inter-particle spacing in Lagrangian space.
    
    * ``sx_full,sy_full,sz_full`` --  3-dim NumPy float32 arrays containing the 
      components of the particle displacements today as calculated in 
      the ZA in the full box. These particles should cover the COLA 
      volume only. If a refined subvolume is provided, these crude 
      particles which reside inside that subvolume are discarded and 
      replaced with the fine particles. There arrays are overwritten.
       
    * ``sx2_full,sy2_full,sz2_full`` -- same as above but for the second 
      order displacement field.
    
    * ``FULL`` -- a boolean (default: ``False``). If True, it indicates to 
      the code that the COLA volume covers the full box. In that case, 
      LPT in the COLA volume is not calculated, as that matches the LPT 
      in the full box.


    * ``cellsize_zoom`` -- a float (default: ``0``). The inter-particle 
      spacing in Lagrangian space for the refined subvolume, if such is 
      provided. If not, ``cellsize_zoom`` must be set to zero 
      (default), as that is used as a check for the presence of that 
      subvolume.
    
    * ``s*_full_zoom,s*2_full_zoom`` -- same as without ``_zoom`` 
      above, but for the refined region (default: ``None``).

    
    * ``offset_zoom`` -- a 3-vector of floats (default: ``None``). Gives the 
      physical coordinates of the origin of the refinement region 
      relative to the the origin of the full box.
    
    * ``BBox_in`` -- a 3x2 array of integers (default: ``None``). It has the 
      form ``[[i0,i1],[j0,j1],[k0,k1]]``, which gives the bounding box 
      for the refinement region in units of the crude particles 
      Lagrangian index. Thus, the particles with displacements 
      ``sx_full|sy_full|sz_full[i0:i1,j0:j1,k0:k1]`` are replaced with the fine 
      particles with displacements ``sx_full_zoom|sy_full_zoom|sz_full_zoom``.

    * ``ngrid_x,ngrid_y,ngrid_z`` -- integers (default: ``None``). Provide the size of the 
      PM grid, which the algorithm uses to calculate the forces for the 
      Kicks.
    
    * ``gridcellsize`` --float (default: ``None``). Provide the grid spacing of the PM 
      grid, which the algorithm uses to calculate the forces for the 
      Kicks.
      
    * ``ngrid_x_lpt,ngrid_y_lpt,ngrid_z_lpt,gridcellsize_lpt`` -- the 
      same as without ``_lpt`` above but for calculating the LPT 
      displacements in the COLA volume. These better match their 
      counterparts above for the force calculation, as mismatches often 
      lead to unexpected non-cancellations and artifacts.
    
    * ``Om`` -- a float (default: ``0.274``), giving the matter density, :math:`\Omega_m`, today.
    
    * ``Ol`` -- a float (default: ``1.-0.274``), giving the vacuum density, :math:`\Omega_\Lambda`, today.

    * ``a_initial`` -- a float (default: ``1./15.``). The initial scale 
      factor from which to start the COLA evolution. This better be 
      near ``1/n_steps``.
      
    * ``a_final`` -- a float (default: ``1.0``). The final scale 
      factor for the COLA evolution.
           
    * ``n_steps`` -- an integer (default: ``15``). The total number of 
      timesteps which the COLA algorithm should make.

    * ``nCola`` -- a float (default: ``-2.5``). The spectral index for the time-domain COLA. 
      Sane values lie in the range ``(-4,3.5)``. Cannot be ``0``, but of course can be near ``0`` (say ``0.001``).
      See Section A.3 of [temporalCOLA]_.
           
    * ``save_to_file`` -- a boolean (default: ``False``). Whether to save the final snapshot to file.
           
    * ``file_npz_out`` -- a string (default: ``'tmp.npz'``), giving the 
      filename for the ``.npz``  `SciPy 
      file <http://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html>`_,
      in which to save the snapshot. See the source file for what is actually saved.
        
    **Return**:
    
    * ``px,py,pz`` -- 3-dim float32 arrays containing the components of the particle positions inside the COLA volume.
    
    * ``vx,vy,vz`` -- 3-dim float32 arrays containing the components of 
      the particle velocities, :math:`\bm{v}`. Velocities are in units 
      of :math:`\mathrm{Mpc}/h` and are calculated according to:
      
      .. math::
        :nowrap:
      
        \begin{eqnarray}
            \bm{v}\equiv \frac{1}{a\,H(a)}\frac{d\bm{x}}{d\eta}
        \end{eqnarray}
    
      where :math:`\eta` is conformal time; :math:`a` is the final scale 
      factor ``a_final``; :math:`H(a)` is the Hubble parameter; 
      :math:`\bm{x}` is the comoving position. This definition allows 
      calculating redshift-space positions trivial: one simply has to 
      add the line-of-sight velocity to the particle position.
    """
 
    from numpy import array,zeros
    from ic import initial_positions,ic_2lpt_engine
    from growth import _vel_coef,_displ_coef, growth_factor_solution,growth_2lpt,d_growth2
    from potential import  get_phi, initialize_density
    from acceleration import grad_phi_engine
    from scipy import interpolate
    from cic import CICDeposit_3
    
    if (cellsize_zoom==0):
        BBox_in=array([[0,0],[0,0],[0,0]],dtype='int32')
    else:
        offset_zoom=offset_zoom.astype('float32')
    offset=array([0.0,0.0,0.0],dtype='float32')
    

# time-related stuff


    da = (a_final - a_initial)/float(n_steps)

    d=growth_factor_solution(Om,Ol)
    growth   =  interpolate.interp1d(d[:,0].tolist(),d[:,1].tolist(),kind='linear')
    d_growth =  interpolate.interp1d(d[:,0].tolist(),d[:,2].tolist(),kind='linear')

    initial_growth_factor   =     growth(a_initial)
    initial_growth2_factor  =     growth_2lpt(a_initial,initial_growth_factor,Om)
    final_d_growth          =   d_growth(a_final  )
    final_d_growth2         =   d_growth2(a_final,final_d_growth,Om,Ol)
    initial_d_growth          =   d_growth( a_initial  )
    initial_d_growth2         =   d_growth2(a_initial,initial_d_growth,Om,Ol)
    del d_growth

#############
    npart_x,npart_y,npart_z=sx_full.shape
    
    npart_x_zoom=None
    npart_y_zoom=None
    npart_z_zoom=None
    if (cellsize_zoom!=0): 
        npart_x_zoom,npart_y_zoom,npart_z_zoom=sx_full_zoom.shape
    
   
#############

    import time
    start = time.time()



#####################
# Do LPT in COLA box
#####################
    if (FULL):
        # if (COLA box)=(full box), then their lpt's match:
        sx=sx_full
        sy=sy_full
        sz=sz_full
           
        sx2 = sx2_full
        sy2 = sy2_full
        sz2 = sz2_full
        
        if (cellsize_zoom!=0): 
            sx_zoom = sx_full_zoom
            sy_zoom = sy_full_zoom
            sz_zoom = sz_full_zoom
               
            sx2_zoom = sx2_full_zoom
            sy2_zoom = sy2_full_zoom
            sz2_zoom = sz2_full_zoom
        
    else:
        # if (COLA box) != (full box), then we need the lpt in the COLA box:
        print "Calculate LPT in the COLA box..."
        sx,sy,sz,sx2,sy2,sz2,sx_zoom,sy_zoom,sz_zoom,sx2_zoom,sy2_zoom,sz2_zoom  = ic_2lpt_engine(   
                sx_full,          
                sy_full,          
                sz_full,          
                cellsize,          
                
                                   
                ngrid_x_lpt,ngrid_y_lpt,ngrid_z_lpt,          
                gridcellsize_lpt,   
                
                with_2lpt=True,          
                sx2_full = sx2_full,             
                sy2_full = sy2_full,             
                sz2_full = sz2_full,             
                          
                cellsize_zoom = cellsize_zoom,             
                BBox_in = BBox_in,             
                sx_full_zoom =  sx_full_zoom,             
                sy_full_zoom =  sy_full_zoom,          
                sz_full_zoom =  sz_full_zoom,
                sx2_full_zoom = sx2_full_zoom,
                sy2_full_zoom = sy2_full_zoom,
                sz2_full_zoom = sz2_full_zoom,
                
                offset_zoom=offset_zoom)
        print "... done"




#######################
# Some initializations:
#######################

# density:
    density,den_k,den_fft,phi_fft  = initialize_density(ngrid_x,ngrid_y,ngrid_z)

# positions:
    print "Initializing particle positions..."
    px, py, pz = initial_positions( 
                                    sx_full, 
                                    sy_full, 
                                    sz_full,
                                    
                                    sx2_full, 
                                    sy2_full, 
                                    sz2_full,
                                    
                                    cellsize, initial_growth_factor, 
                                    initial_growth2_factor,
                                    
                                    ngrid_x,
                                    ngrid_y,
                                    ngrid_z,
                                    gridcellsize)
    if (cellsize_zoom!=0): 
        px_zoom, py_zoom, pz_zoom = initial_positions( 
                                    sx_full_zoom, 
                                    sy_full_zoom, 
                                    sz_full_zoom,
                                    
                                    sx2_full_zoom, 
                                    sy2_full_zoom, 
                                    sz2_full_zoom,
                                    
                                    cellsize_zoom, initial_growth_factor, 
                                    initial_growth2_factor,
                                    
                                   
                                    ngrid_x,
                                    ngrid_y,
                                    ngrid_z,
                                    gridcellsize,
                                    offset = offset_zoom)
    print "...done"
    
# velocities:

    # Initial residual velocities are zero in COLA. 
    # This corresponds to the L_- operator in 1301.0322.
    # But to avoid short-scale 
    # noise, we do the smoothing trick explained in the new paper. 
    # However, that smoothing should not affect the IC velocities!
    # So, first add the full vel, then further down subtract the same but smoothed.
    # 
    # This smoothing is not really needed if FULL=True. But that case is 
    # not very interesting here, so we do it just the same.
    vx = initial_d_growth *  (sx_full) + initial_d_growth2 * (sx2_full)
    vy = initial_d_growth *  (sy_full) + initial_d_growth2 * (sy2_full)
    vz = initial_d_growth *  (sz_full) + initial_d_growth2 * (sz2_full)
    
    if (cellsize_zoom!=0): 
        vx_zoom = initial_d_growth *  (sx_full_zoom) + initial_d_growth2 * (sx2_full_zoom)
        vy_zoom = initial_d_growth *  (sy_full_zoom) + initial_d_growth2 * (sy2_full_zoom)
        vz_zoom = initial_d_growth *  (sz_full_zoom) + initial_d_growth2 * (sz2_full_zoom)
    
    print "Smoothing arrays for the COLA game ..."
    from box_smooth import box_smooth
    tmp=zeros(sx_full.shape,dtype='float32')
    box_smooth(sx_full, tmp)
    sx_full[:]=tmp[:]
    box_smooth(sy_full, tmp)
    sy_full[:]=tmp[:]
    box_smooth(sz_full, tmp)
    sz_full[:]=tmp[:]
    box_smooth(sx2_full, tmp)
    sx2_full[:]=tmp[:]
    box_smooth(sy2_full, tmp)
    sy2_full[:]=tmp[:]
    box_smooth(sz2_full, tmp)
    sz2_full[:]=tmp[:]
    #
    box_smooth(sx, tmp)
    sx[:]=tmp[:]
    box_smooth(sy, tmp)
    sy[:]=tmp[:]
    box_smooth(sz, tmp)
    sz[:]=tmp[:]
    box_smooth(sx2, tmp)
    sx2[:]=tmp[:]
    box_smooth(sy2, tmp)
    sy2[:]=tmp[:]
    box_smooth(sz2, tmp)
    sz2[:]=tmp[:]
    del tmp
    if (cellsize_zoom!=0): 
        tmp=zeros(sx_full_zoom.shape,dtype='float32')
        box_smooth(sx_full_zoom, tmp)
        sx_full_zoom[:]=tmp[:]      
        box_smooth(sy_full_zoom, tmp)
        sy_full_zoom[:]=tmp[:]      
        box_smooth(sz_full_zoom, tmp)
        sz_full_zoom[:]=tmp[:]
        box_smooth(sx2_full_zoom, tmp)
        sx2_full_zoom[:]=tmp[:]
        box_smooth(sy2_full_zoom, tmp)
        sy2_full_zoom[:]=tmp[:]
        box_smooth(sz2_full_zoom, tmp)
        sz2_full_zoom[:]=tmp[:]
        #
        box_smooth(sx_zoom, tmp)
        sx_zoom[:]=tmp[:]      
        box_smooth(sy_zoom, tmp)
        sy_zoom[:]=tmp[:]      
        box_smooth(sz_zoom, tmp)
        sz_zoom[:]=tmp[:]         
        box_smooth(sx2_zoom, tmp)
        sx2_zoom[:]=tmp[:]      
        box_smooth(sy2_zoom, tmp)
        sy2_zoom[:]=tmp[:]      
        box_smooth(sz2_zoom, tmp)
        sz2_zoom[:]=tmp[:]
        del tmp
    print "... done"
    
    #All s* arrays are now smoothed!
    #Next subtract smoothed vels as prescribed above.
    vx -= initial_d_growth *  (sx_full) + initial_d_growth2 * (sx2_full)
    vy -= initial_d_growth *  (sy_full) + initial_d_growth2 * (sy2_full)
    vz -= initial_d_growth *  (sz_full) + initial_d_growth2 * (sz2_full)
    if (cellsize_zoom!=0): 
        vx_zoom -= initial_d_growth *  (sx_full_zoom) + initial_d_growth2 * (sx2_full_zoom)
        vy_zoom -= initial_d_growth *  (sy_full_zoom) + initial_d_growth2 * (sy2_full_zoom)
        vz_zoom -= initial_d_growth *  (sz_full_zoom) + initial_d_growth2 * (sz2_full_zoom)

    #vx = zeros(sx.shape,dtype='float32')
    #vy = zeros(sx.shape,dtype='float32')
    #vz = zeros(sx.shape,dtype='float32')
    #if (cellsize_zoom!=0): 
    #    vx_zoom = zeros(sx_zoom.shape,dtype='float32')
    #    vy_zoom = zeros(sx_zoom.shape,dtype='float32')
    #    vz_zoom = zeros(sx_zoom.shape,dtype='float32')

#scale factors:
    # initialize scale factor
    aiKick  = a_initial
    aKick   = a_initial
    aiDrift = a_initial
    aDrift  = a_initial

    #dummy values, to initialize as global
    afKick=0
    afDrift=0

    dummy=0.0# yet another dummy




####################
#DO THE TIMESTEPS!!!
####################

    for i in range(n_steps+1):
        
        if i==0 or i==n_steps:
            afKick = aiKick+da/2.0
        else:
            afKick = aiKick+da
        
        
        ################
        # FORCES!
        ################
        
        # Calculate PM density:
        density.fill(0.0)
        
                
        CICDeposit_3(px,                                                                   
                     py,                                                                   
                     pz,                                                                   
                     px,py,pz, #dummies                                                    
                     density,            
                     cellsize,gridcellsize,                                                
                     0,                                                                
                     dummy,dummy,BBox_in,offset,1)                                           
        if (cellsize_zoom!=0):                                                                        
           CICDeposit_3(px_zoom,                                                                
                     py_zoom,                                                                 
                     pz_zoom,                                                                 
                     px_zoom,py_zoom,pz_zoom, #dummies                                              
                     density,                      
                     cellsize_zoom,gridcellsize,                                            
                     0,                                                                
                     dummy,dummy,array([[0,0],[0,0],[0,0]],dtype='int32'),offset_zoom,1)                                                          

        density-= 1.0
        
        #print "ave den",density.mean(dtype=float64)
        
        # Calculate potential
        get_phi(density, den_k, den_fft, phi_fft, ngrid_x,ngrid_y,ngrid_z, gridcellsize)
        phi = density # density now holds phi, so rename it
      
        ################
        # KICK!
        ################
        beta  = -1.5*aDrift*Om*_vel_coef(aiKick,afKick,aDrift,nCola,Om,Ol)
        d=growth(aDrift)
        Om143=(Om/(Om+(1.0-Om)*aDrift*aDrift*aDrift))**(1./143.)
        
        
        # Note that grad_phi_engine() will subtract the lpt forces in the COLA volume
        # before doing the kick.
        grad_phi_engine( px,py,pz,  vx,vy,vz,  sx,sy,sz, sx2,sy2,sz2,  beta,   beta,                                             
                                npart_x,npart_y,npart_z, phi,ngrid_x,ngrid_y,ngrid_z, cellsize,gridcellsize, d,                     
                                d*d*(1.0+7./3.*Om143),array([0.0,0.0,0.0],dtype='float32'),0)                                                                              
        if (cellsize_zoom!=0):                                                                                                                                            
            grad_phi_engine( px_zoom,py_zoom,pz_zoom,  vx_zoom,vy_zoom,vz_zoom,  sx_zoom,sy_zoom,sz_zoom, sx2_zoom,sy2_zoom,sz2_zoom,  beta,   beta,                                                    
                                npart_x_zoom,npart_y_zoom,npart_z_zoom, phi,ngrid_x,ngrid_y,ngrid_z, cellsize_zoom,gridcellsize, d,                     
                                d*d*(1.0+7./3.*Om143),array([0.0,0.0,0.0],dtype='float32'),0)  
                          

                                                                                                                                    
        del phi                                                                                                                     
                                                                                                                                       
        aKick  = afKick                                                                                                                    
        aiKick = afKick                                                                                                              
                                                                                                                                       
        print "Kicked to a = "+str(aKick)                                                                                                                                   
        ################                                                                                                                                                    
        # DRIFT!                                                                                                                                                            
        ################                                                                                                                                                    
        if i < n_steps:                                                                                                                                                     
            afDrift = aiDrift + da                                                                                                                                          
            alpha =  _displ_coef(aiDrift,afDrift,aKick,nCola,Om,Ol)                                                                                                          
            gamma =  1.0 * (growth(afDrift) - growth(aiDrift))                                                                                                              
            gamma2 =  1.0 * (growth_2lpt(afDrift,growth(afDrift),Om) - growth_2lpt(aiDrift,growth(aiDrift),Om))                                                                    
            
            # Drift, but also add the lpt displacement from the full volume:                                                                                                                                
            px += vx * alpha + sx_full * gamma + sx2_full * gamma2 + float(ngrid_x)*gridcellsize             
            py += vy * alpha + sy_full * gamma + sy2_full * gamma2 + float(ngrid_y)*gridcellsize             
            pz += vz * alpha + sz_full * gamma + sz2_full * gamma2 + float(ngrid_z)*gridcellsize             
                                                                                                                                                                                   
            px %=  float(ngrid_x)*gridcellsize                                                                                                                                         
            py %=  float(ngrid_y)*gridcellsize                                                                                                                                         
            pz %=  float(ngrid_z)*gridcellsize                                                                                                                                         
            
            if (cellsize_zoom!=0): 
                px_zoom += vx_zoom * alpha + sx_full_zoom * gamma + sx2_full_zoom * gamma2 + float(ngrid_x)*gridcellsize             
                py_zoom += vy_zoom * alpha + sy_full_zoom * gamma + sy2_full_zoom * gamma2 + float(ngrid_y)*gridcellsize             
                pz_zoom += vz_zoom * alpha + sz_full_zoom * gamma + sz2_full_zoom * gamma2 + float(ngrid_z)*gridcellsize             
                                                                                                                                                                                      
                px_zoom %=  float(ngrid_x)*gridcellsize                                                                                                                                         
                py_zoom %=  float(ngrid_y)*gridcellsize                                                                                                                                         
                pz_zoom %=  float(ngrid_z)*gridcellsize    
            
            aiDrift = afDrift                                                                                                                                                      
            aDrift  = afDrift
            print "Drifted to a = "+str(aDrift)
        
        

    del den_k, den_fft, phi_fft,density
    
    
    
    
    # Add back LPT velocity to velocity residual
    # This corresponds to the L_+ operator in 1301.0322.
    vx += final_d_growth *  (sx_full) + final_d_growth2 * (sx2_full)
    vy += final_d_growth *  (sy_full) + final_d_growth2 * (sy2_full)
    vz += final_d_growth *  (sz_full) + final_d_growth2 * (sz2_full)
    
    if (cellsize_zoom!=0): 
        vx_zoom += final_d_growth *  (sx_full_zoom) + final_d_growth2 * (sx2_full_zoom)
        vy_zoom += final_d_growth *  (sy_full_zoom) + final_d_growth2 * (sy2_full_zoom)
        vz_zoom += final_d_growth *  (sz_full_zoom) + final_d_growth2 * (sz2_full_zoom)
    
    # Now convert velocities to 
    # v_{rsd}\equiv (ds/d\eta)/(a H(a)):
    from growth import _q_factor
    rsd_fac=a_final/_q_factor(a_final,Om,Ol)
    vx*=rsd_fac
    vy*=rsd_fac
    vz*=rsd_fac
    if (cellsize_zoom!=0): 
        vx_zoom *=rsd_fac
        vy_zoom *=rsd_fac
        vz_zoom *=rsd_fac
    
    end = time.time()
    print "Time elapsed on small box (including IC): "+str(end - start) + " seconds."
    
    if save_to_file:
        from numpy import savez
        savez( file_npz_out ,
            px_zoom=px_zoom,py_zoom=py_zoom,pz_zoom=pz_zoom,
            vx_zoom=vx_zoom,vy_zoom=vy_zoom,vz_zoom=vz_zoom,
            cellsize_zoom=cellsize_zoom,
            px=px,py=py,pz=pz,
            vx=vx,vy=vy,vz=vz,
            cellsize=cellsize,
            ngrid_x=ngrid_x,ngrid_y=ngrid_y,ngrid_z=ngrid_z,
            z_final=1.0/(aDrift)-1.0,z_init=1.0/(a_initial)-1.0,
            n_steps=n_steps,Om=Om,Ol=Ol,nCola=nCola,
            ngrid_x_lpt=ngrid_x_lpt,ngrid_y_lpt=ngrid_y_lpt,ngrid_z_lpt=ngrid_z_lpt,
            gridcellsize=gridcellsize,
            gridcellsize_lpt=gridcellsize_lpt)
    return px,py,pz,vx,vy,vz,px_zoom,py_zoom,pz_zoom,vx_zoom,vy_zoom,vz_zoom
    
