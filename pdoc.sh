#!/bin/bash
docker run --rm -v `pwd`:/pygraphblas -it graphblas/pygraphblas-minimal:v4.0.3 pdoc --html -f -o . pygraphblas
