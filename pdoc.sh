BASE_NAME="notebook"
BASE_PATH="/home/jovyan"

docker run --rm -v `pwd`:/home/joyvan/pygraphblas -it graphblas/pygraphblas-notebook:latest pdoc --html -f -o /home/joyvan/pygraphblas/docs pygraphblas/
