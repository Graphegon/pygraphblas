docker run --rm --user root -e NB_UID=$(id -u) -e NB_GID=$(id -g) -p 8888:8888 \
	   -v `pwd`/pygraphblas:/home/jovyan/pygraphblas  \
	   -v `pwd`/demo:/home/jovyan/demo  \
	   -v `pwd`/demo:/pygraphblas/demo \
	   -w /home/jovyan/demo  \
	   -it graphblas/pygraphblas-notebook:latest
