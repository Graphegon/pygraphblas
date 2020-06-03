if [ $# -eq 0 ]
    then
        echo "Usage: ./build.sh SS_RELEASE BASE_NAME BRANCH [LOCATION PUSH]"
        echo
        echo "Example: ./build.sh v3.2.0 notebook master clone push"
        exit 1
fi

SS_RELEASE=$1
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
    git clone --branch $BRANCH https://github.com/michelp/pygraphblas.git
    cd pygraphblas
fi

docker build -f Dockerfile-${BASE_NAME} -t graphblas/pygraphblas-${BASE_NAME}:${SS_RELEASE} .

if [ "$PUSH" = "push" ]
then
    docker push graphblas/pygraphblas-${BASE_NAME}:${SS_RELEASE}
    docker tag graphblas/pygraphblas-${BASE_NAME}:${SS_RELEASE} graphblas/pygraphblas-${BASE_NAME}:latest
    docker push graphblas/pygraphblas-${BASE_NAME}:latest
fi
