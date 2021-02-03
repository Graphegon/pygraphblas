SS_COMPACT=1 ./docker_build.sh v4.0.3 test minimal
docker run --rm \
       -v `pwd`/pygraphblas:/pygraphblas/pygraphblas \
       -v `pwd`/demo:/pygraphblas/demo \
       -v `pwd`/docs:/docs \
       -v `pwd`/tests:/pygraphblas/tests \
	   -v `pwd`/.coverage:/pygraphblas/.coverage
       graphblas/pygraphblas-minimal:test \
	   bash -c 'coverage run --branch -m pytest'
