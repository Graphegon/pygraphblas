docker run --rm --user root -e NB_UID=$(id -u) -e NB_GID=$(id -g) -p 8888:8888 \
	   -v `pwd`/pygraphblas:/home/jovyan/pygraphblas  \
	   -v `pwd`/demo:/home/jovyan/demo  \
	   -v `pwd`/demo:/pygraphblas/demo \
	   -e MALLOC_MMAP_MAX=16777216 \
	   -e MALLOC_TRIM_THRESHOLD=-1 \
	   -e MALLOC_TOP_PAD=46777216 \
	   -w /home/jovyan/demo  \
	   -it graphblas/pygraphblas-notebook:latest
