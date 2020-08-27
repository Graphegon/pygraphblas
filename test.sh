if [ -n "$1" ]
  then
      docker pull graphblas/pygraphblas-minimal:latest
fi
docker run --rm \
       -v `pwd`/tests:/pygraphblas/tests \
       -v `pwd`/pygraphblas:/pygraphblas/pygraphblas \
       -it graphblas/pygraphblas-minimal \
       pytest --cov=pygraphblas --cov-report=term-missing $@
