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


Welcome to pyCOLA's documentation!
==================================

:Author: Svetlin Tassev
:Version: |release|
:Date: June 26, 2014
:Homepage: `pyCOLA Homepage`_
:Documentation: `PDF Documentation <https://bitbucket.org/tassev/pycola/downloads/pyCOLA.pdf>`_
:License: `GPLv3+ License`_

.. _pyCOLA Homepage: https://bitbucket.org/tassev/pycola
.. _GPLv3+ License: https://www.gnu.org/licenses/gpl-3.0.html


Introduction
------------

pyCOLA is a multithreaded Python/Cython N-body code, implementing the 
Comoving Lagrangian Acceleration (COLA) method in the temporal and 
spatial domains. pyCOLA also implements a novel method to compute 
second-order cosmological initial conditions for given initial 
conditions at first-order for arbitrary initial particle configurations 
(including glass initial conditions, as well as initial conditions 
having refined subregions).

pyCOLA is based on the following two papers: [temporalCOLA]_, 
[spatialCOLA]_. We kindly ask you [#f1]_ to acknowledge them and their 
authors in any program or publication in which you use the COLA method 
in the temporal and/or spatial domains. 

The new method for calculating second-order cosmological initial 
conditions is based on the following paper: (todo: Daniel, let me know 
what to cite). Again, we kindly ask you to acknowledge that paper and its 
authors in any program or publication in which you use that method.


.. rubric:: Footnotes

.. [#f1] We do not *require* you, however, as we want pyCOLA to be 
   GPLv3 compatible.

pyCOLA requires `NumPy <http://www.numpy.org/>`_, `SciPy 
<http://www.scipy.org/>`_, `pyFFTW 
<https://hgomersall.github.io/pyFFTW/index.html>`_, `h5py 
<http://www.h5py.org/>`_, as well as their respective dependencies. 
Note that pyFFTW v0.9.2 does not support large arrays, so one needs to 
install the development version from `github 
<https://github.com/hgomersall/pyFFTW>`_, where the bug has been fixed.



.. note::
   All lengthscales are in units of comoving :math:`\mathrm{Mpc}/h`, unless 
   otherwise specified.

.. todo::
   If there is interest in the code (i.e. not only the algorithm), it 
   should be converted to use classes as that will enormously reduce 
   the amount of arguments to be passed around, will make the code more 
   readable, and reduce the chances for introducing bugs. Some of the 
   functions are already converted to using classes in a separate 
   branch, but converting the whole code will take some time.  

pyCOLA modules
=================

.. include:: ic.rst
.. include:: growth.rst
.. include:: cic.rst
.. include:: potential.rst
.. include:: acceleration.rst
.. include:: evolve.rst
Auxiliary
-------------------
.. include:: box_smooth.rst
.. include:: aux.rst




Worked-out example
==================

.. include:: example.rst



.. [MUSIC] `Multi-scale initial conditions for cosmological 
   simulations`, O. Hahn, T. Abel, Monthly Notices of the Royal 
   Astronomical Society, 415, 2101 (2011), `arXiv:1103.6031 
   <http://http://arxiv.org/abs/1103.6031>`_. The code can be found 
   on `this website <http://www.phys.ethz.ch/~hahn/MUSIC/>`_.
    
.. [spatialCOLA] `Extending the N-body Comoving Lagrangian 
   Acceleration Method to the Spatial Domain`, S. Tassev, D. 
   J. Eisenstein, B. D. Wandelt, M. Zaldarriaga, (2014), 
   `arXiv:14??.???? <http://arxiv.org/abs/arXiv:14??.????>`_
   
.. [temporalCOLA] `Solving Large Scale Structure in Ten Easy Steps with 
   COLA`, S. Tassev, M. Zaldarriaga, D. J. Eisenstein, Journal of 
   Cosmology and Astroparticle Physics, 06, 036 
   (2013), `arXiv:1301.0322 <http://arxiv.org/abs/arXiv:1301.0322>`_



