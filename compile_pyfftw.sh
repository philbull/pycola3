#!/bin/bash
# Install deps for pyfftw and then compile and install manually
yum -y install fftw-devel wget
wget https://github.com/pyFFTW/pyFFTW/archive/refs/tags/v0.12.0.tar.gz -O pyfftw.tar.gz
tar -xzf pyfftw.tar.gz
cd pyFFTW-0.12.0/
python setup.py build_ext --inplace
python setup.py install
python -c "import pyfftw"
