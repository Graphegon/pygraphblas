from pygraphblas.matrix import Matrix

def test_matrix_create_from_type():
    m = Matrix.from_type(int)
    assert m.nrows == 0
    assert m.ncols == 0
    assert m.nvals == 0
    m = Matrix.from_type(int, 10, 10)
    assert m.nrows == 10
    assert m.ncols == 10
    assert m.nvals == 0

def test_matrix_set_element():
    m = Matrix.from_type(int, 10, 10)
    m[3,3] = 3
    assert m.nrows == 10
    assert m.ncols == 10
    assert m.nvals == 1
    assert m[3,3] == 3

def test_matrix_create_dup():
    m = Matrix.from_type(int, 10, 10)
    m[3,3] = 3
    n = Matrix.dup(m)
    assert n.nrows == 10
    assert n.ncols == 10
    assert n.nvals == 1
    assert n[3,3] == 3
    
def test_matrix_from_edgelists():
    v = Matrix.from_edgelists(
        list(range(10)),
        list(range(10)),
        list(range(10)))
    assert v.nrows == 10
    assert v.ncols == 10
    assert v.nvals == 10

def test_matrix_eq():
    v = Matrix.from_edgelists(
        list(range(10)),
        list(range(10)),
        list(range(10)))
    w = Matrix.from_edgelists(
        list(range(10)),
        list(range(10)),
        list(range(10)))
    x = Matrix.from_edgelists(
        list(range(1,11)),
        list(range(1,11)),
        list(range(1,11)), ncols=11, nrows=11)
    assert v == w
    assert v != x
