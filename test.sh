docker run --rm \
       -v `pwd`/tests:/pygraphblas/tests \
       -v `pwd`/pygraphblas:/pygraphblas/pygraphblas \
       -it graphblas/pygraphblas-minimal:test \
       pytest --cov=pygraphblas --cov-report=term-missing $@
