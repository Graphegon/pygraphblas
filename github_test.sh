SS_COMPACT=1 ./docker_build.sh v4.0.3 test minimal
docker run --rm \
       -v `pwd`/pygraphblas:/pygraphblas/pygraphblas \
       -v `pwd`/demo:/pygraphblas/demo \
       -v `pwd`/docs:/docs \
       -v `pwd`/tests:/pygraphblas/tests \
       graphblas/pygraphblas-minimal:test \
       pytest --cov=pygraphblas --cov-report=term-missing --cov-branch $@
