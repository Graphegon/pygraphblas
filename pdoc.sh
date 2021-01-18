#!/bin/bash
docker run --rm -v `pwd`:/docs -w /pygraphblas -it graphblas/pygraphblas-minimal:test pdoc --html --template-dir /docs/templates -f -o /docs pygraphblas
