#!/bin/bash
docker run --rm -v `pwd`:/docs -w /pygraphblas -it graphblas/pygraphblas-minimal:test pdoc --html -f -o /docs pygraphblas
