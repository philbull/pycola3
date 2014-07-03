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

if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    from aux import boundaries
    from ic import ic_2lpt,import_music_snapshot
    from evolve import evolve
    from cic import CICDeposit_3
    from potential import initialize_density
    
    # Set up the parameters from the MUSIC ic:

    # Set up according to instructions for 
    # aux.boundaries()
    
    boxsize=100.0 # in Mpc/h
    level=9
    level_zoom=10
    gridscale=3
    
    # Set up according to instructions for 
    # ic.import_music_snapshot()
    level0='09' # should match level above
    level1='10' # should match level_zoom above
    

    # Set how much to cut from the sides of the full box. 
    # This makes the COLA box to be of the following size in Mpc/h:
    # (2.**level-(cut_from_sides[0]+cut_from_sides[1]))/2.**level*boxsize
    
    # This is the full box. Set FULL=True in evolve() below
    #cut_from_sides=[0,0]# 100Mpc/h. 
    #
    # These are the interesting cases:
    #cut_from_sides=[64,64]# 75Mpc/h
    #cut_from_sides=[128,128] # 50Mpc/h
    cut_from_sides=[192,192]  # 25Mpc/h



    sx_full1, sy_full1, sz_full1, sx_full_zoom1, sy_full_zoom1, \
        sz_full_zoom1, offset_from_code \
        = import_music_snapshot("/media/stuff/ohahn-music-116353436ee6/ic_za.hdf5", \
                                boxsize,level0=level0,level1=level1)
    
    NPART_zoom=list(sx_full_zoom1.shape)


    print "Starting 2LPT on full box."
    
    #Get bounding boxes for full box with 1 refinement level for MUSIC.
    BBox_in, offset_zoom, cellsize, cellsize_zoom, \
        offset_index, BBox_out, BBox_out_zoom, \
        ngrid_x, ngrid_y, ngrid_z, gridcellsize  \
        = boundaries(boxsize, level, level_zoom, \
                     NPART_zoom, offset_from_code, [0,0], gridscale)

    sx2_full1, sy2_full1, sz2_full1,  sx2_full_zoom1, \
        sy2_full_zoom1, sz2_full_zoom1 \
        = ic_2lpt( 
            cellsize,
            sx_full1 ,
            sy_full1 ,
            sz_full1 ,
            
            cellsize_zoom=cellsize_zoom,
            sx_zoom = sx_full_zoom1,
            sy_zoom = sy_full_zoom1,
            sz_zoom = sz_full_zoom1,
            
            boxsize=100.00, 
            ngrid_x_lpt=ngrid_x,ngrid_y_lpt=ngrid_y,ngrid_z_lpt=ngrid_z,
                       
            offset_zoom=offset_zoom,BBox_in=BBox_in)





    #Get bounding boxes for the COLA box with 1 refinement level for MUSIC.
    BBox_in, offset_zoom, cellsize, cellsize_zoom, \
        offset_index, BBox_out, BBox_out_zoom, \
        ngrid_x, ngrid_y, ngrid_z, gridcellsize \
        = boundaries(
            boxsize, level, level_zoom, \
            NPART_zoom, offset_from_code, cut_from_sides, gridscale)

    # Trim full-box displacement fields down to COLA volume.
    sx_full       =       sx_full1[BBox_out[0,0]:BBox_out[0,1],  \
                                   BBox_out[1,0]:BBox_out[1,1],  \
                                   BBox_out[2,0]:BBox_out[2,1]]
    sy_full       =       sy_full1[BBox_out[0,0]:BBox_out[0,1],  
                                   BBox_out[1,0]:BBox_out[1,1],  \
                                   BBox_out[2,0]:BBox_out[2,1]]
    sz_full       =       sz_full1[BBox_out[0,0]:BBox_out[0,1],  \
                                   BBox_out[1,0]:BBox_out[1,1],  \
                                   BBox_out[2,0]:BBox_out[2,1]]
    sx_full_zoom  =  sx_full_zoom1[BBox_out_zoom[0,0]:BBox_out_zoom[0,1],  \
                                   BBox_out_zoom[1,0]:BBox_out_zoom[1,1],  \
                                   BBox_out_zoom[2,0]:BBox_out_zoom[2,1]]
    sy_full_zoom  =  sy_full_zoom1[BBox_out_zoom[0,0]:BBox_out_zoom[0,1],  \
                                   BBox_out_zoom[1,0]:BBox_out_zoom[1,1],  \
                                   BBox_out_zoom[2,0]:BBox_out_zoom[2,1]]
    sz_full_zoom  =  sz_full_zoom1[BBox_out_zoom[0,0]:BBox_out_zoom[0,1],  \
                                   BBox_out_zoom[1,0]:BBox_out_zoom[1,1],  \
                                   BBox_out_zoom[2,0]:BBox_out_zoom[2,1]]               
    del sx_full1, sy_full1, sz_full1, sx_full_zoom1, sy_full_zoom1, sz_full_zoom1

    sx2_full       =       sx2_full1[BBox_out[0,0]:BBox_out[0,1],  \
                                     BBox_out[1,0]:BBox_out[1,1],  \
                                     BBox_out[2,0]:BBox_out[2,1]]
    sy2_full       =       sy2_full1[BBox_out[0,0]:BBox_out[0,1],  \
                                     BBox_out[1,0]:BBox_out[1,1],  \
                                      BBox_out[2,0]:BBox_out[2,1]]
    sz2_full       =       sz2_full1[BBox_out[0,0]:BBox_out[0,1],  \
                                     BBox_out[1,0]:BBox_out[1,1],  \
                                     BBox_out[2,0]:BBox_out[2,1]]
    sx2_full_zoom  =  sx2_full_zoom1[BBox_out_zoom[0,0]:BBox_out_zoom[0,1],  \
                                     BBox_out_zoom[1,0]:BBox_out_zoom[1,1],  \
                                     BBox_out_zoom[2,0]:BBox_out_zoom[2,1]]
    sy2_full_zoom  =  sy2_full_zoom1[BBox_out_zoom[0,0]:BBox_out_zoom[0,1],  \
                                     BBox_out_zoom[1,0]:BBox_out_zoom[1,1],  \
                                     BBox_out_zoom[2,0]:BBox_out_zoom[2,1]]
    sz2_full_zoom  =  sz2_full_zoom1[BBox_out_zoom[0,0]:BBox_out_zoom[0,1],  \
                                     BBox_out_zoom[1,0]:BBox_out_zoom[1,1],  \
                                     BBox_out_zoom[2,0]:BBox_out_zoom[2,1]]
    del sx2_full1, sy2_full1, sz2_full1, sx2_full_zoom1, sy2_full_zoom1, sz2_full_zoom1


    print "2LPT on full box is done."
    print "Starting COLA!"



    px, py, pz, vx, vy, vz, \
        px_zoom, py_zoom, pz_zoom, vx_zoom, vy_zoom, vz_zoom \
        = evolve( 
            cellsize,
            sx_full, sy_full, sz_full, 
            sx2_full, sy2_full, sz2_full,
            FULL=False,
            
            cellsize_zoom=cellsize_zoom,
            sx_full_zoom  = sx_full_zoom , 
            sy_full_zoom  = sy_full_zoom , 
            sz_full_zoom  = sz_full_zoom ,
            sx2_full_zoom = sx2_full_zoom,
            sy2_full_zoom = sy2_full_zoom,
            sz2_full_zoom = sz2_full_zoom,
            
            offset_zoom=offset_zoom,
            BBox_in=BBox_in,
            
            ngrid_x=ngrid_x,
            ngrid_y=ngrid_y,
            ngrid_z=ngrid_z,
            gridcellsize=gridcellsize,
            
            ngrid_x_lpt=ngrid_x,
            ngrid_y_lpt=ngrid_y,
            ngrid_z_lpt=ngrid_z,
            gridcellsize_lpt=gridcellsize,
            
            a_final=1.,
            a_initial=1./10.,
            n_steps=10,
            
            save_to_file=False,  # set this to True to output the snapshot to a file
            file_npz_out='tmp.npz',
            )

    del vx_zoom,vy_zoom,vz_zoom
    del vx,vy,vz



    print "Making a figure ..."
    # grid size for figure array
    ngrid=2*512
    # physical size of figure array
    cutsize=12.0#Mpc/h
    
    # offset vector [Mpc/h]:
    com=[ 1.30208333,  1.10677083,  0.944]
    com[0] += offset_zoom[0]+cellsize_zoom * \
        (BBox_out_zoom[0,1]-BBox_out_zoom[0,0])/2.0-cutsize/2.0
    com[1] += offset_zoom[1]+cellsize_zoom * \
        (BBox_out_zoom[1,1]-BBox_out_zoom[1,0])/2.0-cutsize/2.0
    com[2] += offset_zoom[2]+cellsize_zoom * \
        (BBox_out_zoom[2,1]-BBox_out_zoom[2,0])/2.0-cutsize/2.0
    


    density,den_k,den_fft,_ = initialize_density(ngrid,ngrid,ngrid)
    density.fill(0.0)

    # Lay down fine particles on density array with CiC:
    CICDeposit_3(
                     px_zoom-com[0],
                     py_zoom-com[1],
                     pz_zoom-com[2],
                     px_zoom,py_zoom,pz_zoom, #dummies
                     density,
                     
                     cellsize_zoom,
                     cutsize/float(ngrid),
                     0,
                     0,     # dummy
                     0,     # dummy
                     
                     np.array([[0,0],[0,0],[0,0]],dtype='int32'),
                     
                     np.array([0.0,0.0,0.0],dtype='float32'),
                     0)
    
    # Lay down any present crude particles on density array with CiC:
    CICDeposit_3(
                     px-com[0],
                     py-com[1],
                     pz-com[2],
                     px,py,pz, #dummies
                     density,
                     
                     cellsize,
                     cutsize/float(ngrid),
                     0,
                     0,     # dummy
                     0,     # dummy
                     
                     BBox_in,
                     
                     np.array([0.0,0.0,0.0],dtype='float32'),
                     0)

    # make the figure:
    plt.imshow(np.arcsinh((density.mean(axis=2))*np.sinh(1.0)/10.0)**(1./3.),
               vmin=0.0,vmax=1.75,
               interpolation='bicubic',cmap='CMRmap_r')
    plt.axis('off')
    plt.show()

