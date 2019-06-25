from pygraphblas.matrix import Matrix
from pygraphblas.base import lib

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

def test_matrix_from_lists():
    v = Matrix.from_lists(
        list(range(10)),
        list(range(10)),
        list(range(10)))
    assert v.nrows == 10
    assert v.ncols == 10
    assert v.nvals == 10

def test_matrix_eq():
    v = Matrix.from_lists(
        list(range(10)),
        list(range(10)),
        list(range(10)))
    w = Matrix.from_lists(
        list(range(10)),
        list(range(10)),
        list(range(10)))
    x = Matrix.from_lists(
        list(range(1,11)),
        list(range(1,11)),
        list(range(1,11)), ncols=11, nrows=11)
    assert v == w
    assert v != x

def test_matrix_gb_type():
    v = Matrix.from_type(bool, 10)
    assert v.gb_type == lib.GrB_BOOL
    v = Matrix.from_type(int, 10)
    assert v.gb_type == lib.GrB_INT64
    v = Matrix.from_type(float, 10)
    assert v.gb_type == lib.GrB_FP64

def test_vector_ewise_add():
    v = Matrix.from_lists(
        list(range(10)),
        list(range(10)),
        list(range(10)))
    w = Matrix.from_lists(
        list(range(10)),
        list(range(10)),
        list(range(10)))

    x = v.ewise_add(w)
    assert x == Matrix.from_lists(
        list(range(10)),
        list(range(10)),
        list(range(0, 20, 2)))

    z = v & w
    assert x == z

def test_vector_ewise_mult():
    v = Matrix.from_lists(
        list(range(10)),
        list(range(10)),
        list(range(10)))
    w = Matrix.from_lists(
        list(range(10)),
        list(range(10)),
        list(range(10)))

    x = v.ewise_mult(w)
    assert x == Matrix.from_lists(
        list(range(10)),
        list(range(10)),
        list(map(lambda x: x*x, list(range(10)))))

    z = v | w
    assert x == z
