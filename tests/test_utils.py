from pygraphblas import *

def test_add_identity():
    A = Matrix.sparse(INT8, 10, 10)
    assert add_identity(A) == 10
    A = Matrix.sparse(INT8, 10, 10)
    A[5,5] = 42
    assert add_identity(A) == 9
    
