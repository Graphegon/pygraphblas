if [ $# -eq 0 ]
    then
        echo "Usage: ./docker_build.sh PY_RELEASE BASE_NAME BRANCH [LOCATION PUSH]"
        echo
        echo "Example: ./docker_build.sh 5.1.7.1 notebook main clone push"
        exit 1
fi

PY_RELEASE=$1
BASE_NAME=$2
BRANCH=$3
LOCATION=$4
PUSH=$5

if [ "$LOCATION" = "clone" ]
then
    TMPDIR=$(mktemp -d)
    if [ ! -e $TMPDIR ]; then
        >&2 echo "Failed to create temp directory"
        exit 1
    fi
    trap "exit 1"           HUP INT PIPE QUIT TERM
    trap 'rm -rf "$TMPDIR"' EXIT
    
    cd $TMPDIR
    git clone --branch $BRANCH https://github.com/Graphegon/pygraphblas.git
    cd pygraphblas
fi

docker build \
       -f Dockerfile-${BASE_NAME} \
       -t graphblas/pygraphblas-${BASE_NAME}:${PY_RELEASE} \
       .

docker tag graphblas/pygraphblas-${BASE_NAME}:${PY_RELEASE} graphblas/pygraphblas-${BASE_NAME}:latest

if [ "$PUSH" = "push" ]
then
    docker push graphblas/pygraphblas-${BASE_NAME}:${PY_RELEASE}
    docker push graphblas/pygraphblas-${BASE_NAME}:latest
fi
