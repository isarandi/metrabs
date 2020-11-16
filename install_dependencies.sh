#!/usr/bin/env bash
set -euo pipefail

# This was tested on Ubuntu 18.04.5
sudo apt install build-essential --yes wget curl gfortran git ncurses-dev unzip tar

# Anaconda (anaconda.com) is the simplest way to install most of the dependencies.
# If you don't have it installed yet, install Miniconda as follows:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b
eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"

# Create a new environment and install the dependencies:
conda create --yes --name metrabs python=3.8 Cython matplotlib pillow imageio ffmpeg scikit-image scikit-learn tqdm numba cachetools Pillow
conda activate metrabs
conda install --yes opencv -c conda-forge
pip install tensorflow tf_slim tensorflow-addons attrdict jpeg4py imageio-ffmpeg transforms3d more_itertools spacepy

# Install my fork of the COCO tools, used for managing runlength-encoded (RLE) masks.
# The additional functionality in my fork is for mask inversion in RLE, which is only needed for generating the MuCo dataset.
git clone https://github.com/isarandi/cocoapi
cd cocoapi/PythonAPI
make
python setup.py install
cd ../..
rm -rf cocoapi

# We need to install the [CDF library](https://cdf.gsfc.nasa.gov/) because Human3.6M supplies the annotations as cdf files.
# We read them using the [SpacePy](https://spacepy.github.io/) Python library, which in turn depends on the CDF library.
wget https://spdf.sci.gsfc.nasa.gov/pub/software/cdf/dist/cdf37_1/linux/cdf37_1-dist-cdf.tar.gz
tar xf cdf37_1-dist-cdf.tar.gz
rm cdf37_1-dist-cdf.tar.gz
cd cdf37_1-dist
make OS=linux ENV=gnu CURSES=yes FORTRAN=no UCOPTIONS=-O2 SHARED=yes -j4 all

# If you have sudo rights, simply run `sudo make install`. If you have no `sudo` rights, make sure to add the
# `cdf37_1-dist/src/lib` to the `LD_LIBRARY_PATH` environment variable (add to ~/.bashrc for permanent effect), or use GNU Stow.
# The following will work temporarily:
export LD_LIBRARY_PATH=$PWD/src/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Optional:
# Install libjpeg-turbo for faster JPEG decoding.
# wget https://sourceforge.net/projects/libjpeg-turbo/files/2.0.5/libjpeg-turbo-2.0.5.tar.gz
# Then compile it.
# Or use the repo:
# git clone https://github.com/libjpeg-turbo/libjpeg-turbo.git
# cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -G"Unix Makefiles" .
# make
