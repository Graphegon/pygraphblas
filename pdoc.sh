#!/bin/bash
docker run --rm -v `pwd`:/pygraphblas -it graphblas/pygraphblas-minimal:latest pdoc --html -f -o . pygraphblas
