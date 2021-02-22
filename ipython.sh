BASE_NAME=${1:-minimal}
shift
BASE_PATH="/pygraphblas"

if [ "$BASE_NAME" = "notebook" ]
  then
	  BASE_PATH="/home/jovyan"
fi
docker run --rm -v `pwd`/demo:/demo -v ~/GAP/:/GAP -v `pwd`/gap:/gap -v `pwd`/tests:${BASE_PATH}/tests -v `pwd`${BASE_PATH}:/pygraphblas/pygraphblas -it graphblas/pygraphblas-${BASE_NAME}:latest ipython $@
