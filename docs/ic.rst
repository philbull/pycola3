.. ########################################################################
.. ########################################################################
.. #    Copyright (c) 2013,2014       Svetlin Tassev
.. #                       Princeton University,Harvard University
.. #
.. #   This file is part of pyCOLA.
.. #
.. #   pyCOLA is free software: you can redistribute it and/or modify
.. #   it under the terms of the GNU General Public License as published by
.. #   the Free Software Foundation, either version 3 of the License, or
.. #   (at your option) any later version.
.. #
.. #   pyCOLA is distributed in the hope that it will be useful,
.. #   but WITHOUT ANY WARRANTY; without even the implied warranty of
.. #   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
.. #   GNU General Public License for more details.
.. #
.. #   You should have received a copy of the GNU General Public License
.. #   along with pyCOLA.  If not, see <http://www.gnu.org/licenses/>.
.. #
.. ########################################################################
.. ########################################################################

Initial conditions
------------------

First order initial conditions for pyCOLA can be calculated using 
either  MUSIC  [MUSIC]_, or internally. With MUSIC, however, one can 
do refinements on a region, which is not supported internally.

The second-order displacement field is generated using a novel 
algorithm using force evaluations. See the Algorithm section of 
:func:`ic_2lpt_engine` for details.

.. warning::
    As of MUSIC `rev. 116353436ee6 
    <https://bitbucket.org/ohahn/music/commits/116353436ee6ff009abda2e51cd4792a209fa63b>`_, 
    the second-order displacement field returned by MUSIC gets 
    unphysical large-scale deviations when a refined subvolume is 
    requested (seems to be fine for single grid). Until that problem is 
    fixed, use the function :func:`ic.ic_2lpt` to get the second order 
    displacements from the first order result. Update: MUSIC received a 
    fix with `rev. ed51fcaffee 
    <https://bitbucket.org/ohahn/music/commits/ed51fcaffeec08d676b8a3435a0c4009f4024220>`_,
    which supposedly fixes the problem.
        



Initial displacements at first order 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: ic.import_music_snapshot

.. autofunction:: ic.ic_za

Initial displacements at second order 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: ic.ic_2lpt

.. autofunction:: ic.ic_2lpt_engine

Obtaining the Eulerian positions 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: ic.initial_positions


.. automodule:: ic
    :undoc-members:
    :show-inheritance:
