#!/bin/bash
if [ "$1" = "build" ]
then
	SS_COMPACT=1 ./docker_build.sh v4.0.1 test minimal
fi
docker run --rm \
       -v `pwd`/pygraphblas:/pygraphblas/pygraphblas \
       -v `pwd`/demo:/pygraphblas/demo \
       -v `pwd`/docs:/docs \
       -v `pwd`/tests:/pygraphblas/tests \
       -it graphblas/pygraphblas-minimal:test \
       pytest --cov=pygraphblas --cov-report=term-missing $@
