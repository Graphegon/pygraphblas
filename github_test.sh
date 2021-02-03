#SS_COMPACT=1 ./docker_build.sh v4.0.3 test minimal
docker run --rm \
       -v `pwd`/pygraphblas:/pygraphblas/pygraphblas \
       -v `pwd`/demo:/pygraphblas/demo               \
       -v `pwd`/docs:/docs \
       -v `pwd`/tests:/pygraphblas/tests \
	   -e COVERAGE_FILE=/pygraphblas/tests/.coverage \
       graphblas/pygraphblas-minimal:test \
	   coverage run --branch -m pytest
