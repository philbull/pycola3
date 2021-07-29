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

def boundaries(boxsize,level,level_zoom,NPART_zoom,offset_from_code,cut_from_sides, gridscale):
    """ 
    :math:`\\vspace{-1mm}`
    
    Calculate bounding boxes, fine grid offsets and other 
    quantities useful when dealing with a MUSIC snapshot. 
    
    **Arguments**:
    
    * The first three arguments must be set to the following parameters 
      from the MUSIC configuration file::
      
          [boundaries() argument]  [parameter from MUSIC .conf]    [type]
          boxsize                =          boxlength               float
          level                  =          levelmin                  int
          level_zoom             =          levelmax                  int
      
      With the included MUSIC configuration file, :download:`ics.conf 
      <./ics.conf>`, the above three arguments take the values: 
      100.0, 9, 10.
    
    * ``NPART_zoom`` -- list of three integers, giving the size of the 
      fine grid. 
    
    * ``offset_from_code`` -- list of three integers, giving the 
      crude-grid index coordinates of the origin of the fine grid.
      
    * The last two parameters define the COLA volume and the size of the PM grid:
    
        - ``cut_from_sides`` -- a list of two integers, call them :math:`a,b`. 
          Then, the Lagrangian COLA volume in terms of the indices of 
          the particle displacements arrays, ``s``:sub:`i`, at level ``level`` 
          (the one covering the full box) is given by the following 
          array slice::
          
            si[a:2**level-b,a:2**level-b,a:2**level-b]
      
        - ``gridscale`` -- an integer. Sets the size of the PM grid in 
          each dimension to ``gridscale`` times the particle number at 
          level ``level`` in that dimension within the COLA volume.
    
    **Return**:
    
    * ``BBox_in,offset_zoom,cellsize,cellsize_zoom`` -- the same as in 
      :func:`ic.ic_2lpt` but the first two give the bounding box and 
      offset of the refinement region with respect to the COLA volume.
    
    * ``offset_index`` -- a 3-vector of integers. The same as 
      ``offset_zoom`` but in units of the crude-particle index.
    
    * ``BBox_out`` -- a 3x2 array of integers. Gives the bounding box 
      at level ``level`` that resides in the COLA volume. It equals 
      ``[[a,2**level-b],[a,2**level-b],[a,2**level-b]]`` (see the 
      description of the argument ``cut_from_sides`` above).
    
    * ``BBox_out_zoom`` -- the same as ``BBox_out`` but for the fine 
      particles (as level ``level_zoom``) included in the COLA volume.
    
    * ``ngrid_x, ngrid_y, ngrid_z`` -- integers. The size of the PM 
      grid used in the COLA volume.
    
    * ``gridcellsize`` -- a float. The PM grid spacing in 
      :math:`\mathrm{Mpc}/h`.
      
    **Example**: For example usage, see the worked out example in (todo).
    
    """

    from numpy import array
    NPART = (2**level,2**level,2**level)
    cellsize=boxsize/2.0**level
    cellsize_zoom=boxsize/2.0**level_zoom
    gridcellsize=(cellsize)/float(gridscale)
    BBox_out    =  array([[cut_from_sides[0], NPART[0]-cut_from_sides[1]], 
                          [cut_from_sides[0], NPART[1]-cut_from_sides[1]], 
                          [cut_from_sides[0], NPART[2]-cut_from_sides[1]]],dtype='int32')
    ind0=[0,0,0]
    ind1=list(NPART_zoom[:])
    for i in range(3):
        if (cut_from_sides[0]>offset_from_code[i]):
            ind0[i]=(cut_from_sides[0]-offset_from_code[i])*2**(level_zoom-level)
        if (NPART[i]-cut_from_sides[1]-offset_from_code[i])*2**(level_zoom-level) < NPART_zoom[i]:
            ind1[i]=(NPART[i]-cut_from_sides[1]-offset_from_code[i])*2**(level_zoom-level)
        
    BBox_out_zoom  =  array([[ind0[0], ind1[0]], [ind0[1], ind1[1]], [ind0[2], ind1[2]]],dtype='int32')

    #offset index of small sim box relative to large sim box (NOTE: sim, not IC boxes!!!)
    offset_index = array(offset_from_code,dtype='int32')+BBox_out_zoom[:,0]//2**(level_zoom-level)-BBox_out[:,0]
    offset_zoom = offset_index  *  cellsize - array([cellsize_zoom,cellsize_zoom,cellsize_zoom],dtype='float32')/2.0
    


    #bbox of small sim box relative to large sim box in large sim coo
    BBox_in   =  array([[offset_index[0], offset_index[0] + (BBox_out_zoom[0,1]-BBox_out_zoom[0,0])//2**(level_zoom-level)], 
                           [offset_index[1], offset_index[1] + (BBox_out_zoom[1,1]-BBox_out_zoom[1,0])//2**(level_zoom-level)], 
                           [offset_index[2], offset_index[2] + (BBox_out_zoom[2,1]-BBox_out_zoom[2,0])//2**(level_zoom-level)]],dtype='int32')
    ngrid_x=(BBox_out[0,1]-BBox_out[0,0])*gridscale
    ngrid_y=(BBox_out[1,1]-BBox_out[1,0])*gridscale
    ngrid_z=(BBox_out[2,1]-BBox_out[2,0])*gridscale
    return BBox_in,offset_zoom,cellsize,cellsize_zoom,offset_index,BBox_out,BBox_out_zoom,ngrid_x,ngrid_y,ngrid_z,gridcellsize

    
