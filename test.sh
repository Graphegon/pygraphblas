docker run -v `pwd`/tests:/pygraphblas/tests -v `pwd`/pygraphblas:/pygraphblas/pygraphblas -it graphblas/pygraphblas-notebook pytest --cov=pygraphblas --cov-report=term-missing
