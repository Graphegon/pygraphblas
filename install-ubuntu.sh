#!/bin/bash

# Use this script to install pygraphblas on Ubuntu 18.04
set -e

# Install the dependencies for building the C libraries
sudo apt install -y curl m4 g++

# Install the required version of SuiteSparse:GraphBLAS

SS_RELEASE=v4.0.1
SS_BURBLE=0
SS_COMPACT=0

git clone --branch ${SS_RELEASE} --single-branch https://github.com/DrTimothyAldenDavis/GraphBLAS.git
cd GraphBLAS/build
cmake .. -DGB_BURBLE=${SS_BURBLE} -DGBCOMPACT=${SS_COMPACT}
make -j$(nproc)
sudo make install
cd ../..

# Install Python components
sudo apt install -y python3-pip
pip3 install -r notebook-requirements.txt

# Install pygraphblas as a user (does not require root privileges)
python3 setup.py install --user
