docker build --build-arg SS_RELEASE=$1 . -t graphblas/pygraphblas-notebook:$1
docker tag graphblas/pygraphblas-notebook graphblas/pygraphblas-notebook:latest
