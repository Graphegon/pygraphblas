mkdir -p dnn_demo && cd dnn_demo

if [ ! -f "sparse-images-1024.tsv" ]; then
    wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/mnist/sparse-images-1024.tsv.gz
    gunzip sparse-images-1024.tsv.gz
    rm sparse-images-1024.tsv.gz
fi

if [ ! -d "neuron1024" ]; then
    wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron1024.tar.gz
    zcat neuron1024.tar.gz | tar xvf -
    rm neuron1024.tar.gz
fi

cd ..

docker run -v `pwd`/dnn_demo:/pygraphblas/dnn_demo -v `pwd`/pygraphblas:/pygraphblas/pygraphblas -it pygraphblas/pygraphblas ipython
