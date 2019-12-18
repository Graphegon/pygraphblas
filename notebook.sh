docker run --user root -e NB_UID=$(id -u) -e NB_GID=$(id -g) -p 8888:8888 -v `pwd`/pygraphblas/demo:/home/jovyan/pygraphblas/demo -it graphblas/pygraphblas-notebook
