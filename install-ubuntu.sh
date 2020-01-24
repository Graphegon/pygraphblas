#!/bin/bash

# Use this script to install pygraphblas on Ubuntu 18.04

# Install the dependencies for building the C libraries
sudo apt install -y curl m4 g++

# Install the required versions of SuiteSparse:GraphBLAS and LAGraph
export JOBS=$(nproc) # use all threads for compiling
curl -s -L  http://faculty.cse.tamu.edu/davis/GraphBLAS/GraphBLAS-3.1.1.tar.gz | tar -xz && \
    cd GraphBLAS-3.1.1 && \
    make library && \
    sudo make install && \
    sudo ldconfig && \
    cd ..
git clone --branch 22July2019 https://github.com/GraphBLAS/LAGraph.git && \
    cd LAGraph && \
    make library && \
    sudo make install && \
    sudo ldconfig && \
    cd ..

# Install Python components
sudo apt install -y python3-pip
pip3 install setuptools pytest pytest-cov ipdb RISE graphviz numba contextvars colorama

# Install pygraphblas as a user (does not require root privileges)
python3 setup.py install --user
