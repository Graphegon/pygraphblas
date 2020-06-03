if [ $# -eq 0 ]
    then
        echo "Usage: ./docker_build.sh SS_RELEASE BASE_NAME BRANCH [LOCATION PUSH]"
        echo "Note: BASE_NAME=minimal image does not use SS_RELEASE parameter"
        echo
        echo "Example: ./docker_build.sh v3.2.0 notebook master clone push"
        exit 1
fi

SS_RELEASE=$1
BASE_NAME=$2
BRANCH=$3
LOCATION=$4
PUSH=$5

# for BASE_NAME=notebook image
# set env var to 1 for faster SuiteSparse compilation, but the code will be slower
SS_COMPACT=${SS_COMPACT:-0}

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
    git clone --branch $BRANCH https://github.com/michelp/pygraphblas.git
    cd pygraphblas
fi

docker build \
       --build-arg SS_RELEASE=${SS_RELEASE} \
       --build-arg SS_COMPACT=${SS_COMPACT} \
       -f Dockerfile-${BASE_NAME} \
       -t graphblas/pygraphblas-${BASE_NAME}:${SS_RELEASE} \
       .

docker tag graphblas/pygraphblas-${BASE_NAME}:${SS_RELEASE} graphblas/pygraphblas-${BASE_NAME}:latest

if [ "$PUSH" = "push" ]
then
    docker push graphblas/pygraphblas-${BASE_NAME}:${SS_RELEASE}
    docker push graphblas/pygraphblas-${BASE_NAME}:latest
fi
