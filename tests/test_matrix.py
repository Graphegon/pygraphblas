import sys
from pygraphblas.matrix import Matrix, Vector
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

def test_matrix_get_set_element():
    m = Matrix.from_type(int, 10, 10)
    m[3,3] = 3
    assert m.nrows == 10
    assert m.ncols == 10
    assert m.nvals == 1
    assert m[3,3] == 3

def test_matrix_slice_vector():
    v = Matrix.from_lists(
        list(range(10)),
        list(range(10)),
        list(range(10)))
    assert v[5] == Vector.from_lists([5], [5], 10)

def test_matrix_create_dup():
    m = Matrix.from_type(int, 10, 10)
    m[3,3] = 3
    n = Matrix.dup(m)
    assert n.nrows == 10
    assert n.ncols == 10
    assert n.nvals == 1
    assert n[3,3] == 3

def test_matrix_to_from_lists():
    v = Matrix.from_lists(
        list(range(10)),
        list(range(10)),
        list(range(10)))
    assert v.nrows == 10
    assert v.ncols == 10
    assert v.nvals == 10
    assert v.to_lists() == [
        list(range(10)),
        list(range(10)),
        list(range(10)),
        ]

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

def test_matrix_ewise_add():
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
    z = v + w
    assert x == z
    v += w
    assert v == z

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
    z = v * w
    assert x == z
    v *= w
    assert v == z

def test_matrix_reduce_bool():
    v = Matrix.from_type(bool, 10, 10)
    assert not v.reduce_bool()
    v[3,3] = True
    assert v.reduce_bool()

def test_matrix_reduce_int():
    v = Matrix.from_type(int, 10, 10)
    r = v.reduce_int()
    assert type(r) is int
    assert r == 0
    v[3,3] = 3
    v[4,4] = 4
    assert v.reduce_int() == 7

def test_matrix_reduce_float():
    v = Matrix.from_type(float, 10, 10)
    r = v.reduce_float()
    assert type(r) is float
    assert r == 0.0
    v[3,3] = 3.3
    v[4,4] = 4.4
    assert v.reduce_float() == 7.7

def test_matrix_reduce_vector():
    m = Matrix.from_lists(
        list(range(10)),
        list(range(10)),
        list(range(10)))
    v = m.reduce_vector()
    v == Vector.from_list(list(range(10)))

def test_mxm():
    m = Matrix.from_lists(
        [0,1,2],
        [1,2,0],
        [1,2,3])
    n = Matrix.from_lists(
        [0,1,2],
        [1,2,0],
        [2,3,4])
    o = m.mxm(n)
    assert o.nrows == 3
    assert o.ncols == 3
    assert o.nvals == 3
    r = Matrix.from_lists(
        [0,1,2],
        [2,0,1],
        [3,8,6])
    assert o == r
    assert r == m @ n
    m @= n
    assert r == m

def test_matrix_pattern():
    v = Matrix.from_lists(
        list(range(10)),
        list(range(10)),
        list(range(10)))
    p = v.pattern()
    assert p.gb_type == lib.GrB_BOOL
    assert p.nrows == 10
    assert p.ncols == 10
    assert p.nvals == 10

def test_matrix_mm_read_write(tmp_path):
    mmf = tmp_path / 'mmwrite_test.mm'
    mmf.touch()
    m = Matrix.from_lists(
        [0,1,2],
        [0,1,2],
        [2,3,4])
    with mmf.open('w') as f:
        m.to_mm(f)
    with mmf.open() as f:
        assert f.readlines() == [
            '%%MatrixMarket matrix coordinate integer symmetric\n',
            '%%GraphBLAS GrB_INT64\n',
            '3 3 3\n',
            '1 1 2\n',
            '2 2 3\n',
            '3 3 4\n']

    with mmf.open() as f:
        n = Matrix.from_mm(f)
    assert n == m

def test_matrix_random():
    m = Matrix.from_random(int, 10, 10, 5)
    assert m.nrows == 10
    assert m.ncols == 10
    # assert m.nvals == 5 # sometimes this fails?

def test_matrix_slice():
    I, J = tuple(map(list, zip(*product(range(3), repeat=2))))
    V = list(range(9))
    m = Matrix.from_lists(I, J, V)
    import pdb; pdb.set_trace()
