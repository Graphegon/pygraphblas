#!/bin/bash
#SS_COMPACT=1 ./docker_build.sh v4.0.1 test minimal
docker run --rm \
       -v `pwd`/tests:/pygraphblas/tests \
       -v `pwd`/pygraphblas:/pygraphblas/pygraphblas \
       -it graphblas/pygraphblas-minimal:test \
       pytest --cov=pygraphblas --cov-report=term-missing $@
