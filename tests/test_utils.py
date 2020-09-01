from pygraphblas import *


def test_get_version():
    v = get_version()
    assert isinstance(v, tuple)
    assert isinstance(v[0], int)
    assert isinstance(v[1], int)
