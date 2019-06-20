from pygraphblas.vector import Vector

def test_vector_create_from_type():
    m = Vector.from_type(int)
    assert m.size == 0
    assert m.nvals == 0
    m = Vector.from_type(int, 10)
    assert m.size == 10

def test_vector_set_element():
    m = Vector.from_type(int, 10)
    m[3] = 3
    assert m.size == 10
    assert m.nvals == 1
    assert m[3] == 3

def test_vector_create_dup():
    m = Vector.from_type(int, 10)
    m[3] = 3
    n = Vector.dup(m)
    assert n.size == 10
    assert n.nvals == 1
    assert n[3] == 3
