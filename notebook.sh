docker run --user root -e NB_UID=$(id -u) -e NB_GID=$(id -g) -p 8888:8888 -v `pwd`/pygraphblas:/home/jovyan/pygraphblas -it graphblas/pygraphblas-notebook:latest
