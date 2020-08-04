docker run --rm -v ~/GAP/:/GAP -v `pwd`/gap:/gap -v `pwd`/tests:/pygraphblas/tests -v `pwd`/pygraphblas:/pygraphblas/pygraphblas -it graphblas/pygraphblas-minimal:latest ipython $@
