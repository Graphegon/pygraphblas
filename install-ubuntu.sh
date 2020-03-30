#!/bin/bash

# Use this script to install pygraphblas on Ubuntu 18.04
set -e

# Install the dependencies for building the C libraries
sudo apt install -y curl m4 g++

# Install the required version of SuiteSparse:GraphBLAS

SS_RELEASE=v3.2.0
SS_BURBLE=0
JOBS=$(nproc) # use all threads for compiling

git clone  --branch ${SS_RELEASE} --single-branch https://github.com/DrTimothyAldenDavis/GraphBLAS.git
cd GraphBLAS
make library CFLAGS=-DGB_BURBLE=${SS_BURBLE}
sudo make install
cd ..

# Install Python components
sudo apt install -y python3-pip
pip3 install -r notebook-requirements.txt

# Install pygraphblas as a user (does not require root privileges)
python3 setup.py install --user
