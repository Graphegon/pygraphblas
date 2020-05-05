if [ -n "$1" ]
  then
      docker pull graphblas/pygraphblas-minimal:latest
fi
docker run --rm -v `pwd`/tests:/home/jovyan/tests -v `pwd`/pygraphblas:/home/jovyan/pygraphblas -it graphblas/pygraphblas-minimal pytest --cov=pygraphblas --cov-report=term-missing
