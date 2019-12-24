
import pytest
from pygraphblas import *
from pygraphblas.base import lib

def test_scalar_create_from_type():

    m = Scalar.from_type(INT64)
    assert m.nvals == 0
    assert not m
    with pytest.raises(KeyError) as e:
        assert m[0]

    m[0] = 2
    assert m[0] == 2
    assert m.nvals == 1
    assert m

def test_scalar_from_value():

    m = Scalar.from_value(2)
    assert m[0] == 2
    assert m.nvals == 1
    assert m

def test_scalar_dup():

    n = Scalar.from_value(2)
    m = Scalar.dup(n)
    assert m[0] == 2
    assert m.nvals == 1
    assert m

def test_scalar_clear():

    m = Scalar.from_value(2)
    assert m
    assert m[0] == 2
    assert m.nvals == 1
    m.clear()
    assert not m
    assert m.nvals == 0
    with pytest.raises(KeyError) as e:
        assert m[0]
