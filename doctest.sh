#!/bin/bash
if [ "$1" = "build" ]
then
	SS_COMPACT=1 ./docker_build.sh v4.0.1 test minimal
fi
docker run --rm \
       -v `pwd`/pygraphblas:/pygraphblas/pygraphblas \
       -v `pwd`/docs:/docs \
       -it graphblas/pygraphblas-minimal:test \
       python3 -c 'import pygraphblas; pygraphblas.run_doctests()'
