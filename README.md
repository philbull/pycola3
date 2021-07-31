[![Documentation Status](https://readthedocs.org/projects/pycola3/badge/?version=latest)](https://pycola3.readthedocs.io/en/latest/?badge=latest)

Author: Svetlin Tassev

Ported to Python 3 by [Phil Bull](http://philbull.com/)

Website: [pycola3](https://github.com/philbull/pycola3) | [Original pyCOLA](https://bitbucket.org/tassev/pycola)

Documentation (old version): [PDF Documentation](https://github.com/philbull/pycola3/files/6911023/pyCOLA.pdf)

License: [GPLv3+ License](https://www.gnu.org/licenses/gpl-3.0.html)

``pycola3`` is a multithreaded Python/Cython N-body code, implementing the
Comoving Lagrangian Acceleration (COLA) method in the temporal and
spatial domains.

``pycola3`` is based on the following two papers:

1. Solving Large Scale Structure in Ten Easy Steps with
   COLA, S. Tassev, M. Zaldarriaga, D. J. Eisenstein, Journal of
   Cosmology and Astroparticle Physics, 06, 036
   (2013), [arXiv:1301.0322](http://arxiv.org/abs/arXiv:1301.0322)

2. sCOLA: The N-body COLA Method Extended to the Spatial Domain, S. Tassev, D.
   J. Eisenstein, B. D. Wandelt, M. Zaldarriaga, (2015)

If you use ``pycola3`` or the COLA method in the spatial and/or time domains for
scientific work, we kindly ask you to reference the papers above.

* ``pycola3`` is free and open-source software, distributed under the GPLv3 license.

* Currently, the best way to install ``pycola3`` is to clone this git repository and run the setup script like so:

```
  python setup.py install
```

If you are using Python 3.9, you may need to install ``pyfftw3`` using ``conda`` first, before attempting to run the ``setup.py`` script.

While there is a ``pip`` package for ``pycola3``, it is currently not functional due to dependency/binary compatibility issues.


``pycola3`` depends on the following packages:
 * numpy
 * scipy
 * cython
 * pyfftw
 * h5py
 * multiprocessing


Read the manual [here](https://bitbucket.org/tassev/pycola/downloads/pyCOLA.pdf).
