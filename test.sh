#!/bin/bash
python3 -m pytest --cov=pygraphblas --cov-report=term-missing --cov-branch $@
