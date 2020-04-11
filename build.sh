if [ $# -eq 0 ]
    then
        echo "Usage: ./build.sh SS_RELEASE"
        echo
        echo "Example: ./build.sh v3.2.0"
        exit 1
fi

docker build --build-arg SS_RELEASE=$1 -t graphblas/pygraphblas-notebook:$1 .
docker tag graphblas/pygraphblas-notebook:$1 graphblas/pygraphblas-notebook:latest
