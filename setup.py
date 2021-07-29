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

from setuptools import setup, Extension, find_namespace_packages
from Cython.Build import cythonize

import numpy as np
import os

sources = {'pycola._cic':          'pycola/src/cic.pyx',
           'pycola._potential':    'pycola/src/potential.pyx',
           'pycola._acceleration': 'pycola/src/acceleration.pyx',
           'pycola._box_smooth':   'pycola/src/box_smooth.pyx'
          }

# Compile and link arguments for Cython modules
extra_compile_args = ['-fopenmp', '-O3', '-pthread', '-fPIC', '-fwrapv',
                      '-fno-strict-aliasing']
extra_link_args = ['-fopenmp', '-O3', '-pthread', '-fPIC', '-fwrapv', 
                   '-fno-strict-aliasing']

include_dirs = [np.get_include()] # get numpy includes
library_dirs = []
package_data = {}
libraries = [] 

# Define Cython files as ext modules
ext_modules = []
for module in sources.keys():
    ext = Extension(module, [sources[module]],
                    extra_compile_args=extra_compile_args,
                    extra_link_args=extra_link_args,
                    libraries=libraries,
                    include_dirs=include_dirs
                   )
    ext_modules.append(ext)


long_description="""pyCOLA is a multithreaded Python/Cython N-body code, 
implementing the Comoving Lagrangian Acceleration (COLA) method in the 
temporal and spatial domains. 

pyCOLA is based on the following two papers:

1. Solving Large Scale Structure in Ten Easy Steps with
   COLA, S. Tassev, M. Zaldarriaga, D. J. Eisenstein, Journal of
   Cosmology and Astroparticle Physics, 06, 036
   (2013), [arXiv:1301.0322](http://arxiv.org/abs/arXiv:1301.0322)

2. sCOLA: The N-body COLA Method Extended to the Spatial Domain, S. Tassev, D.
   J. Eisenstein, B. D. Wandelt, M. Zaldarriaga, (2015)

Please cite them if using this code for scientific research.

pyCOLA requires `NumPy <http://www.numpy.org/>_`, `SciPy 
<http://www.scipy.org/>`_, `pyFFTW 
<https://hgomersall.github.io/pyFFTW/index.html>`_, `h5py 
<http://www.h5py.org/>`_. Note that pyFFTW v0.9.2 does not support 
large arrays, so one needs to install the development version from 
`github <https://github.com/hgomersall/pyFFTW>`_, where the bug has 
been fixed.

The pyCOLA documentation can be found 
`here <https://bitbucket.org/tassev/pycola/downloads/pyCOLA.pdf>`_, and the source
is on `bitbucket <https://bitbucket.org/tassev/pycola>`_.
"""


setup_args = {
        'name': 'pycola',
        'version': 3.0,
        'author': 'Svetlin Tassev',
        'author_email': 'stassev@alum.mit.edu',
        'description': 'A Python/Cython N-body code, implementing the Comoving Lagrangian Acceleration (COLA) method in the temporal and spatial domains.',
        'url': 'http://ascl.net/1509.007',
        'long_description': long_description,
        'package_dir': {"pycola": "pycola"},
        'packages': find_namespace_packages(),
        'classifiers': [
            'Programming Language :: Python',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Cython',
            'Development Status :: 5 - Production/Stable',
            'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Astronomy',
            'Topic :: Scientific/Engineering :: Physics'
            ],
        'ext_modules': cythonize(ext_modules, language_level=3),
        'include_dirs': include_dirs,
        "install_requires": [
            "numpy",
            "scipy",
            "h5py",
            "pyfftw",
            "multiprocessing",
        ],
  }


if __name__ == '__main__':
    setup(**setup_args)
