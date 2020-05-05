if [ $# -eq 0 ]
    then
        echo "Usage: ./build.sh SS_RELEASE BASE_NAME BRANCH"
        echo
        echo "Example: ./build.sh v3.2.0 minimal master"
        exit 1
fi

SS_RELEASE=$1
BASE_NAME=$2
BRANCH=$3

/bin/rm -Rf docker_build
mkdir docker_build
pushd docker_build
git clone --branch $BRANCH https://github.com/michelp/pygraphblas.git
cd pygraphblas
docker build -f Dockerfile-${BASE_NAME} --build-arg SS_RELEASE=${SS_RELEASE} -t graphblas/pygraphblas-${BASE_NAME}:${SS_RELEASE} .
docker push graphblas/pygraphblas-${BASE_NAME}:${SS_RELEASE}
docker tag graphblas/pygraphblas-${BASE_NAME}:${SS_RELEASE} graphblas/pygraphblas-${BASE_NAME}:latest
docker push graphblas/pygraphblas-${BASE_NAME}:latest

