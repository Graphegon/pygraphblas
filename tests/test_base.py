import pytest

from pygraphblas import *


def test_options_set():
    options_set(nthreads=4)
    options_set(chunk=4096)
    options_set(burble=1)
