docker run -v `pwd`/tests:/pygraphblas/tests -v `pwd`/pygraphblas:/pygraphblas/pygraphblas -it pygraphblas/pygraphblas pytest --cov=pygraphblas --cov-report=term-missing
