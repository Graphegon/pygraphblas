SS_COMPACT=1 ./docker_build.sh v4.0.3 test minimal
docker run --rm \
	   -e GITHUB_TOKEN=${GITHUB_TOKEN} \
	   -e COVERALLS_FLAG_NAME=${COVERALLS_FLAG_NAME} \
	   -e COVERALLS_PARALLEL=${COVERALLS_PARALLEL} \
       -v `pwd`/pygraphblas:/pygraphblas/pygraphblas \
       -v `pwd`/demo:/pygraphblas/demo \
       -v `pwd`/docs:/docs \
       -v `pwd`/tests:/pygraphblas/tests \
       graphblas/pygraphblas-minimal:test \
	   bash -c 'coverage run --branch -m pytest; coveralls'
