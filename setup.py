#This script is based on setup.py included in the pyFFTW 
#package under the BSD license. That still requires the copyright 
#notice below the gpl.

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
#
#  This file incorporates work covered by the following copyright and  
#  permission notice:  
#
#       Copyright 2014 Knowledge Economy Developments Ltd
#       
#       Henry Gomersall
#       heng@kedevelopments.co.uk
#       
#       All rights reserved.
#       
#       Redistribution and use in source and binary forms, with or without
#       modification, are permitted provided that the following conditions are met:
#       
#       * Redistributions of source code must retain the above copyright notice, this
#       list of conditions and the following disclaimer.
#       
#       * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#       
#       * Neither the name of the copyright holder nor the names of its contributors
#       may be used to endorse or promote products derived from this software without
#       specific prior written permission.
#       
#       THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#       AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#       IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#       ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#       LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#       CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#       SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#       INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#       CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#       ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#       POSSIBILITY OF SUCH DAMAGE.
#

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from distutils.util import get_platform
from distutils.ccompiler import get_default_compiler

import os

try:
    from Cython.Distutils import build_ext as build_ext


    sources = [os.path.join(os.getcwd(), 'cic.pyx'),
               os.path.join(os.getcwd(), 'potential.pyx'),
               os.path.join(os.getcwd(), 'acceleration.pyx'),
               os.path.join(os.getcwd(), 'box_smooth.pyx')
    ]
except ImportError as e:
    sources = [os.path.join(os.getcwd(), 'cic.c'),
               os.path.join(os.getcwd(), 'potential.c'),
               os.path.join(os.getcwd(), 'acceleration.c'),
               os.path.join(os.getcwd(), 'box_smooth.c')
    ]
    for i in sources:
        if not os.path.exists(i):
            print i
            print os.path.exists(i)
            raise ImportError(str(e) + '. ' +
                'Cython is required to build the initial .c file.')

    # We can't cythonize, but that's ok as it's been done already.
    from distutils.command.build_ext import build_ext



include_dirs = []
library_dirs = []
package_data = {}

libraries = [] 


ext_modules = [Extension(
    "cic",
    [sources[0]],
    extra_compile_args=['-fopenmp','-O3','-shared' ,'-pthread' ,'-fPIC' ,'-fwrapv','-fno-strict-aliasing'],
    extra_link_args=['-fopenmp','-O3','-shared' ,'-pthread' ,'-fPIC' ,'-fwrapv','-fno-strict-aliasing'],
    libraries=libraries,
    include_dirs=include_dirs
),
Extension(
    "potential",
    [sources[1]],
    extra_compile_args=['-fopenmp','-O3','-shared' ,'-pthread' ,'-fPIC' ,'-fwrapv','-fno-strict-aliasing'],
    extra_link_args=['-fopenmp','-O3','-shared' ,'-pthread' ,'-fPIC' ,'-fwrapv','-fno-strict-aliasing'],
    libraries=libraries,
    include_dirs=include_dirs
),
Extension(
    "acceleration",
    [sources[2]],
    extra_compile_args=['-fopenmp','-O3','-shared' ,'-pthread' ,'-fPIC' ,'-fwrapv','-fno-strict-aliasing'],
    extra_link_args=['-fopenmp','-O3','-shared' ,'-pthread' ,'-fPIC' ,'-fwrapv','-fno-strict-aliasing'],
    libraries=libraries,
    include_dirs=include_dirs
),
Extension(
    "box_smooth",
    [sources[3]],
    extra_compile_args=['-fopenmp','-O3','-shared' ,'-pthread' ,'-fPIC' ,'-fwrapv','-fno-strict-aliasing'],
    extra_link_args=['-fopenmp','-O3','-shared' ,'-pthread' ,'-fPIC' ,'-fwrapv','-fno-strict-aliasing'],
    libraries=libraries,
    include_dirs=include_dirs
)
]

long_description="""pyCOLA is a multithreaded Python/Cython N-body code, 
implementing the Comoving Lagrangian Acceleration (COLA) method in the 
temporal and spatial domains. 

pyCOLA is based on the following two papers:
(todo)
...
...
please, cite them if using for scientific research.

pyCOLA requires `NumPy <http://www.numpy.org/>_`, `SciPy 
<http://www.scipy.org/>`_, `pyFFTW 
<https://hgomersall.github.io/pyFFTW/index.html>`_, `h5py 
<http://www.h5py.org/>`_. Note that pyFFTW v0.9.2 does not support 
large arrays, so one needs to install the development version from 
`github <https://github.com/hgomersall/pyFFTW>`_, where the bug has 
been fixed.

The pyCOLA documentation can be found 
`here <https://bitbucket.org/tassev/pycola/.pdf>`_, and the source
is on `bitbucket <https://bitbucket.org/tassev/pycola>`_.
"""


setup_args = {
        'name': 'pyCOLA',
        'version': 1.0,
        'author': 'Svetlin Tassev',
        'author_email': 'stassev@alum.mit.edu',
        'description': 'A Python/Cython N-body code, implementing the Comoving Lagrangian Acceleration (COLA) method in the temporal and spatial domains.',
        'url': '',
        'long_description': long_description,
        'classifiers': [
            'Programming Language :: Python',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Cython',
            'Development Status :: 5 - Production/Stable',
            'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Astronomy',
            'Topic :: Scientific/Engineering :: Physics'
            ],
        'packages':['ic','evolve','acceleraton', 'cic','potential','box_smooth'],
        'ext_modules': ext_modules,
        'include_dirs': include_dirs,
        'cmdclass' : {'build_ext': build_ext},
  }



if __name__ == '__main__':
    setup(**setup_args)
