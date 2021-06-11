download () {
    if [ ! -f "sparse-images-$2.tsv" ]; then
        wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/mnist/sparse-images-$2.tsv.gz
        gunzip sparse-images-$2.tsv.gz
        rm sparse-images-$2.tsv.gz
    fi

    if [ ! -f "neuron$2-l$3-categories.tsv" ]; then
        wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron$2-l$3-categories.tsv
    fi

    if [ ! -d "neuron$2" ]; then
        wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron$2.tar.gz
        zcat neuron$2.tar.gz | tar xf -
        rm neuron$2.tar.gz
    fi
}

mkdir -p $1 && cd $1

neurons=( 1024 4096 16384 65536 )
nlayers=( 120 480 1920 )

if [ $# -eq 1 ]; then
    for i in "${neurons[@]}"
    do
        for j in "${nlayers[@]}"
        do
            download $1 $i $j
        done
    done
else
    download $1 $2 $3
fi

cd -

docker run --rm --env DEST=/$1 --env NEURONS=$2 --env NLAYERS=$3 \
       -v `pwd`/$1:/$1 \
       -v `pwd`:/pygraphblas \
       -it graphblas/pygraphblas-minimal:test ipython -i -m demo.dnn
