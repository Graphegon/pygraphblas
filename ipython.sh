docker run --rm -v ~/GAP/:/GAP -v `pwd`/tests:/home/jovyan/tests -v `pwd`/pygraphblas:/home/jovyan/pygraphblas -it graphblas/pygraphblas-notebook:latest ipython $@
