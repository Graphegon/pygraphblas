mkdir -p dnn_demo && cd dnn_demo

NNEURONS=$1
NLAYERS=$2

if [ ! -f "sparse-images-$NNEURONS.tsv" ]; then
    wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/mnist/sparse-images-$NNEURONS.tsv.gz
    gunzip sparse-images-$NNEURONS.tsv.gz
    rm sparse-images-$NNEURONS.tsv.gz
fi

if [ ! -f "neuron$NNEURONS-l$NLAYERS-categories.tsv" ]; then
    wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron$NNEURONS-l$NLAYERS-categories.tsv
fi

if [ ! -d "neuron$NNEURONS" ]; then
    wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron$NNEURONS.tar.gz
    zcat neuron$NNEURONS.tar.gz | tar xvf -
    rm neuron$NNEURONS.tar.gz
fi

cd ..

docker run --env NNEURONS=$NNEURONS --env NLAYERS=$NLAYERS -v `pwd`/dnn_demo:/pygraphblas/dnn_demo -v `pwd`/pygraphblas:/pygraphblas/pygraphblas -it pygraphblas/pygraphblas ipython -i -m pygraphblas.demo.dnn

