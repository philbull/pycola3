**Author:** Svetlin V. Tassev (Harvard U, Princeton U)

Ported to Python 3 by Phil Bull (Queen Mary University of London)

*Initial public release date:* Jul 3, 2014

*Python 3 port released:* 29 Jul 2021

``pycola`` is a multithreaded Python/Cython N-body code, implementing the
Comoving Lagrangian Acceleration (COLA) method in the temporal and
spatial domains.

``pycola`` is based on the following two papers:

1. Solving Large Scale Structure in Ten Easy Steps with
   COLA, S. Tassev, M. Zaldarriaga, D. J. Eisenstein, Journal of
   Cosmology and Astroparticle Physics, 06, 036
   (2013), [arXiv:1301.0322](http://arxiv.org/abs/arXiv:1301.0322)

2. sCOLA: The N-body COLA Method Extended to the Spatial Domain, S. Tassev, D.
   J. Eisenstein, B. D. Wandelt, M. Zaldarriaga, (2015)

If you use ``pycola`` or the COLA method in the spatial and/or time domains for
scientific work, we kindly ask you to reference the papers above.

* ``pycola`` is free and open-source software, distributed under the GPLv3 license.

* To build and install the code, you can either use ``pip``:

```
pip install git+https://github.com/philbull/pycola3.git
```

or, alternatvely, install locally from a git checkout:

```
  python setup.py install
```

``pycola`` depends on the following packages:
 * numpy
 * scipy
 * cython
 * pyfftw
 * h5py
 * multiprocessing


Read the manual [here](https://bitbucket.org/tassev/pycola/downloads/pyCOLA.pdf).
