#!/bin/bash
if [ "$1" = "build" ]
then
	SS_COMPACT=1 ./docker_build.sh master test minimal
fi
docker run --rm \
       -v `pwd`/pygraphblas:/pygraphblas/pygraphblas \
       -v `pwd`/demo:/pygraphblas/demo \
       -v `pwd`/docs:/docs \
       -v `pwd`/tests:/pygraphblas/tests \
       -it graphblas/pygraphblas-minimal:test \
       python3 -m pytest --cov=pygraphblas --cov-report=term-missing --cov-branch $@
