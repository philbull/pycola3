Author: Svetlin V. Tassev (Harvard U, Princeton U)
Initial public release date: Jul 3,2014

pyCOLA is a multithreaded Python/Cython N-body code, implementing the 
Comoving Lagrangian Acceleration (COLA) method in the temporal and 
spatial domains.

pyCOLA is based on the following two papers:

1. Solving Large Scale Structure in Ten Easy Steps with 
   COLA, S. Tassev, M. Zaldarriaga, D. J. Eisenstein, Journal of 
   Cosmology and Astroparticle Physics, 06, 036 
   (2013), `arXiv:1301.0322 <http://arxiv.org/abs/arXiv:1301.0322>`_

2. Extending the N-body Comoving Lagrangian 
   Acceleration Method to the Spatial Domain, S. Tassev, D. 
   J. Eisenstein, B. D. Wandelt, M. Zaldarriaga, (2014)

If you use pyCOLA or the COLA method in the spatial and/or time domains for scientific work, we kindly ask you to reference the papers above.

* pyCOLA is free and open-source software, distributed under the GPLv3 license.

* To build the code, you need to run::
  
  python setup.py build_ext --inplace

* To compile successfully, you need to have the following packages installed: `Python 2.7 <https://www.python.org/>`_, `Cython <http://cython.org/>`_, `NumPy <http://www.numpy.org/>`_, `SciPy 
<http://www.scipy.org/>`_, `pyFFTW 
<https://hgomersall.github.io/pyFFTW/index.html>`_, `h5py 
<http://www.h5py.org/>`_, as well as their respective dependencies. 
Note that pyFFTW v0.9.2 does not support large arrays, so one needs to 
install the development version from `github 
<https://github.com/hgomersall/pyFFTW>`_, where the bug has been fixed.

* Read the manual at ???
