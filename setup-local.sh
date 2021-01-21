#!/bin/bash

git clone https://github.com/Graphegon/pygraphblas.git --depth 1 --branch main
cd pygraphblas

python3 -m virtualenv --python=python3 venv
. venv/bin/activate

RUN pip3 install -r minimal-requirements.txt
RUN python3 setup.py install
