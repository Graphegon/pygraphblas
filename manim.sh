docker run --rm -w /tmp/input -v `pwd`/manim/input:/tmp/input -v `pwd`/manim/output:/tmp/output -it graphblas/pygraphblas-manim:latest manim --media_dir=/tmp/output $@
