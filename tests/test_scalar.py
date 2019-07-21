
from pygraphblas import Scalar
from pygraphblas.base import lib

def test_vector_create_from_type():
    
    m = Scalar.from_type(int)
    assert m.nvals == 0
    assert not m
    
    m[0] = 2
    assert m[0] == 2
    assert m.nvals == 1
    assert m
