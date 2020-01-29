import sys
from operator import mod
from itertools import product, repeat

import pytest

from pygraphblas import *
from pygraphblas.base import lib

def test_matrix_create_from_type():
    m = Matrix.from_type(INT8)
    assert m.nrows == 0
    assert m.ncols == 0
    assert m.nvals == 0
    m = Matrix.from_type(INT8, 10, 10)
    assert m.nrows == 10
    assert m.ncols == 10
    assert m.nvals == 0
    assert len(m) == 0

def test_matrix_get_set_element():
    m = Matrix.from_type(INT8, 10, 10)
    m[3,3] = 3
    assert m.nrows == 10
    assert m.ncols == 10
    assert m.nvals == 1
    assert len(m) == 1
    assert m[3,3] == 3

def test_matrix_slice_vector():
    v = Matrix.from_lists(
        list(range(10)),
        list(range(10)),
        list(range(10)))
    assert v[5] == Vector.from_lists([5], [5], 10)

def test_clear():
    v = Matrix.from_lists(
        list(range(10)),
        list(range(10)),
        list(range(10)))
    assert v.nvals == 10
    assert len(v) == 10
    v.clear()
    assert v.nvals == 0
    assert len(v) == 0

def test_resize():
    v = Matrix.from_lists(
        list(range(10)),
        list(range(10)),
        list(range(10)))
    assert v.nrows == 10
    assert v.ncols == 10
    assert v.nvals == 10
    v.resize(20, 20)
    assert v.nrows == 20
    assert v.ncols == 20
    assert v.nvals == 10
    assert list(v.rows) == list(range(10))
    assert list(v.cols) == list(range(10))
    assert list(v.vals) == list(range(10))

def test_matrix_create_dup():
    m = Matrix.from_type(INT8, 10, 10)
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

def test_matrix_gb_type():
    v = Matrix.from_type(BOOL, 10)
    assert v.gb_type == lib.GrB_BOOL
    v = Matrix.from_type(INT8, 10)
    assert v.gb_type == lib.GrB_INT8
    v = Matrix.from_type(FP64, 10)
    assert v.gb_type == lib.GrB_FP64

def test_matrix_eadd():
    v = Matrix.from_lists(
        list(range(10)),
        list(range(10)),
        list(range(10)))
    w = Matrix.from_lists(
        list(range(10)),
        list(range(10)),
        list(range(10)))

    x = v.eadd(w)
    assert x.iseq(Matrix.from_lists(
        list(range(10)),
        list(range(10)),
        list(range(0, 20, 2))))
    z = v + w
    assert x == z
    v += w
    assert v == z

def test_vector_emult():
    v = Matrix.from_lists(
        list(range(10)),
        list(range(10)),
        list(range(10)))
    w = Matrix.from_lists(
        list(range(10)),
        list(range(10)),
        list(range(10)))

    x = v.emult(w)
    assert x.iseq(Matrix.from_lists(
        list(range(10)),
        list(range(10)),
        list(map(lambda x: x*x, list(range(10))))))
    z = v * w
    assert x.iseq(z)
    v *= w
    assert v.iseq(z)

def test_matrix_reduce_bool():
    v = Matrix.from_type(BOOL, 10, 10)
    assert not v.reduce_bool()
    v[3,3] = True
    assert v.reduce_bool()

def test_matrix_reduce_int():
    v = Matrix.from_type(INT8, 10, 10)
    r = v.reduce_int()
    assert type(r) is int
    assert r == 0
    v[3,3] = 3
    v[4,4] = 4
    assert v.reduce_int() == 7

def test_matrix_reduce_float():
    v = Matrix.from_type(FP64, 10, 10)
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
    assert o.iseq(r)
    assert r.iseq(m @ n)
    m @= n
    assert r.iseq(m)
    o = m.mxm(n, semiring=semiring.lor_land_bool)
    assert o.iseq(Matrix.from_lists(
        [0, 1, 2],
        [0, 1, 2],
        [1, 1, 1]))

    with semiring.plus_plus:
        o = m @ n

    assert o.iseq(Matrix.from_lists(
        [0, 1, 2],
        [0, 1, 2],
        [7, 10, 9]))
    
    with semiring.lor_land_bool:
        o = m @ n
    assert o.iseq(Matrix.from_lists(
        [0, 1, 2],
        [0, 1, 2],
        [1, 1, 1]))
    with semiring.lor_land_bool:
        o = m @ n
    assert o.iseq(Matrix.from_lists(
        [0, 1, 2],
        [0, 1, 2],
        [1, 1, 1]))
    with pytest.raises(TypeError):
        m @ 3
    with pytest.raises(TypeError):
        m @ Scalar.from_value(3)

def test_mxv():
    m = Matrix.from_lists(
        [0,1,2],
        [1,2,0],
        [1,2,3])
    v = Vector.from_lists(
        [0,1,2],
        [2,3,4])
    o = m.mxv(v)
    assert o == Vector.from_lists(
        [0, 1, 2],
        [3, 8, 6])

    assert m @ v == o

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

def test_matrix_transpose():
    v = Matrix.from_lists(
        list(range(2, -1, -1)),
        list(range(3)),
        list(range(3)))
    w = v.transpose()
    assert w.iseq(Matrix.from_lists(
        [0, 1, 2],
        [2, 1, 0],
        [0, 1, 2]
        ))

#@pytest.mark.skip
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
        n = Matrix.from_mm(f, INT8)
    assert n.iseq(m)

pytest.mark.skip()
def test_matrix_tsv_read(tmp_path):
    mmf = tmp_path / 'tsv_test.mm'
    mmf.touch()
    with mmf.open('w') as f:
        f.writelines([
            '3\t3\t3\n',
            '1\t1\t2\n',
            '2\t2\t3\n',
            '3\t3\t4\n'])

    with mmf.open() as f:
        n = Matrix.from_tsv(f, INT8, 3, 3)
    assert n.to_lists() == [
        [0, 1, 2],
        [0, 1, 2],
        [2, 3, 4]]

def test_matrix_random():
    m = Matrix.from_random(INT8, 10, 10, 5, seed=42)
    assert m.nrows == 10
    assert m.ncols == 10
    assert len(list(m)) == 5
    m = Matrix.from_random(INT8, 10, 10, 5)
    assert m.nrows == 10
    assert m.ncols == 10
    assert 0 < len(list(m)) <= 5

def test_matrix_slicing():
    I, J = tuple(map(list, zip(*product(range(3), repeat=2))))
    V = list(range(9))
    m = Matrix.from_lists(I, J, V, 3, 3)
    v = m[2]
    assert v == Vector.from_lists(
        [0, 1, 2],
        [6, 7, 8])

    # slice out row vector
    v = m[2,:]
    assert v == Vector.from_lists(
        [0, 1, 2],
        [6, 7, 8])

    # slice out column vector
    v = m[:,2]
    assert v == Vector.from_lists(
        [0, 1, 2],
        [2, 5, 8])

    # slice copy
    n = m[:]
    assert n.iseq(m)
    # also slice copy
    n = m[:,:]
    assert n.iseq(m)

    # submatrix slice out rows
    sm = m[0:1]
    assert sm.iseq(Matrix.from_lists(
        [0, 0, 0, 1, 1, 1],
        [0, 1, 2, 0, 1, 2],
        [0, 1, 2, 3, 4, 5], 2, 3))

    # submatrix slice out columns
    n = m[:,1:]
    assert n.iseq(Matrix.from_lists(
        [0, 0, 1, 1, 2, 2],
        [0, 1, 0, 1, 0, 1],
        [1, 2, 4, 5, 7, 8], 3, 2))

    # submatrix slice out column range
    sm = m[:,1:2]
    assert sm.iseq(Matrix.from_lists(
        [0, 0, 1, 1, 2, 2],
        [0, 1, 0, 1, 0, 1],
        [1, 2, 4, 5, 7, 8], 3, 2))

    # submatrix slice out rows
    n = m[1:,:]
    assert n.iseq(Matrix.from_lists(
        [0, 0, 0, 1, 1, 1],
        [0, 1, 2, 0, 1, 2],
        [3, 4, 5, 6, 7, 8], 2, 3))

    # submatrix slice out row range
    sm = m[1:2,:]
    assert sm.iseq(Matrix.from_lists(
        [0, 0, 0, 1, 1, 1],
        [0, 1, 2, 0, 1, 2],
        [3, 4, 5, 6, 7, 8], 2, 3))

def test_matrix_assign():
    m = Matrix.from_lists(
        list(range(3)),
        list(range(3)),
        list(range(3)))
    assert m.nvals == 3

    m[2] = Vector.from_list(list(repeat(6, 3)))
    assert m.nvals == 5
    assert m.iseq(Matrix.from_lists(
        [0, 1, 2, 2, 2],
        [0, 1, 0, 1, 2],
        [0, 1, 6, 6, 6], 3, 3))

    m = Matrix.from_lists(
        list(range(3)),
        list(range(3)),
        list(range(3)))
    assert m.nvals == 3

    m[2,:] = Vector.from_list(list(repeat(6, 3)))
    assert m.nvals == 5
    assert m.iseq(Matrix.from_lists(
        [0, 1, 2, 2, 2],
        [0, 1, 0, 1, 2],
        [0, 1, 6, 6, 6], 3, 3))

    m = Matrix.from_lists(
        list(range(3)),
        list(range(3)),
        list(range(3)))

    assert m.nvals == 3
    m[:,2] = Vector.from_list(list(repeat(6, 3)))
    assert m.nvals == 5
    assert m.iseq(Matrix.from_lists(
        [0, 1, 0, 1, 2],
        [0, 1, 2, 2, 2],
        [0, 1, 6, 6, 6], 3, 3))

    m = Matrix.from_type(INT64, 3, 3)
    assert m.nvals == 0
    n = Matrix.from_lists(
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2])
    m[:,:] = n
    assert m.iseq(n)

    n = Matrix.from_lists(
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2])

    mask = m > 0
    m[mask] = 9
    assert m.iseq(Matrix.from_lists(
        [0, 1, 2],
        [0, 1, 2],
        [0, 9, 9]))

def test_kron():
    n = Matrix.from_lists(
        list(range(3)),
        list(range(3)),
        list(range(3)))
    m = Matrix.from_lists(
        list(range(3)),
        list(range(3)),
        list(range(3)))

    o = n.kron(m)
    assert o.iseq(Matrix.from_lists(
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [0, 0, 0, 0, 1, 2, 0, 2, 4]))

def test_apply():
    v = Matrix.from_lists(
        [0, 1, 2],
        [0, 1, 2],
        [2, 3, 4])
    w = v.apply(unaryop.ainv_int64)
    assert w.iseq(Matrix.from_lists(
        [0, 1, 2],
        [0, 1, 2],
        [-2, -3, -4]))

def test_apply_lambda():
    v = Matrix.from_lists(
        [0, 1, 2],
        [0, 1, 2],
        [22, 33, 44])

    w = v.apply(lambda x: mod(x, 10))
    assert w.iseq(Matrix.from_lists(
        [0, 1, 2],
        [0, 1, 2],
        [2, 3, 4]))

    w = v.apply(lambda x: mod(x, 7))
    assert w.iseq(Matrix.from_lists(
        [0, 1, 2],
        [0, 1, 2],
        [1, 5, 2]))

def test_get_set_options():
    v = Matrix.from_random(INT8, 10, 10, 10, seed=42)
    v.options_set(hyper=lib.GxB_ALWAYS_HYPER, format=lib.GxB_BY_COL)
    assert v.options_get() == (1.0, lib.GxB_BY_COL, True)

def test_select():
    v = Matrix.from_lists(
        [0, 1, 2],
        [0, 1, 2],
        [0, 0, 3])
    w = v.select(lib.GxB_NONZERO)
    assert w.to_lists() == [[2], [2], [3]]

    w = v.select('!=0')
    assert w.to_lists() == [[2], [2], [3]]

    w = v.select('!=', 0)
    assert w.to_lists() == [[2], [2], [3]]

    w = v.select('>', 0)
    assert w.to_lists() == [[2], [2], [3]]

    w = v.select('<', 3)
    assert w.to_lists() == [[0, 1], [0, 1], [0, 0]]

    w = v.select('>=', 0)
    assert w.iseq(v)

    w = v.select('>=0')
    assert w.iseq(v)

def test_select_ops():
    I, J = tuple(map(list, zip(*product(range(3), repeat=2))))
    V = list(range(9))
    m = Matrix.from_lists(I, J, V, 3, 3)

    assert m.tril().iseq(Matrix.from_lists(
        [0, 1, 1, 2, 2, 2],
        [0, 0, 1, 0, 1, 2],
        [0, 3, 4, 6, 7, 8]))

    assert m.triu().iseq(Matrix.from_lists(
        [0, 0, 0, 1, 1, 2],
        [0, 1, 2, 1, 2, 2],
        [0, 1, 2, 4, 5, 8]))

    assert m.diag().iseq(Matrix.from_lists(
        [0, 1, 2],
        [0, 1, 2],
        [0, 4, 8]))

    assert m.offdiag().iseq(Matrix.from_lists(
        [0, 0, 1, 1, 2, 2],
        [1, 2, 0, 2, 0, 1],
        [1, 2, 3, 5, 6, 7]))

    assert m.nonzero().iseq(Matrix.from_lists(
        [0, 0, 1, 1, 1, 2, 2, 2],
        [1, 2, 0, 1, 2, 0, 1, 2],
        [1, 2, 3, 4, 5, 6, 7, 8]))

    assert (-m).iseq(Matrix.from_lists(
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [0, -1, -2, -3, -4, -5, -6, -7, -8]))

    n = -m

    assert abs(m).iseq(Matrix.from_lists(
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [0, 1, 2, 3, 4, 5, 6, 7, 8]))

    m = Matrix.from_lists(
        [0, 1, 2],
        [0, 1, 2],
        [0.0, 1.0, 2.0], 3, 3)

    n = ~m
    assert n.iseq(Matrix.from_lists(
        [0, 1, 2],
        [0, 1, 2],
        [float('inf'), 1.0, 0.5]))

def test_cmp_scalar():
    I, J = tuple(map(list, zip(*product(range(3), repeat=2))))
    V = list(range(9))
    m = Matrix.from_lists(I, J, V, 3, 3)

    n = m > 5
    assert n == Matrix.from_lists(
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [False, False, False, False, False, False, True, True, True],
        )

    n = m >= 5
    assert n == Matrix.from_lists(
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [False, False, False, False, False, True, True, True, True],
        )

    n = m < 5
    assert n == Matrix.from_lists(
        [0, 0, 0, 1, 1],
        [0, 1, 2, 0, 1],
        [True, True, True, True, True],
        3, 3
        )

    n = m <= 5
    assert n == Matrix.from_lists(
        [0, 0, 0, 1, 1, 1],
        [0, 1, 2, 0, 1, 2],
        [True, True, True, True, True, True],
        3, 3
        )

    n = m == 5
    assert n == Matrix.from_lists(
        [1], [2], [5],
        3, 3
        )

    n = m != 5
    assert n == Matrix.from_lists(
        [0, 0, 0, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 0, 1, 2],
        [0, 1, 2, 3, 4, 6, 7, 8],
        3, 3
        )

def test_cmp():
    I, J = tuple(map(list, zip(*product(range(3), repeat=2))))
    V = list(range(9))
    W = [0, 1, 2, 4, 5, 6, 7, 8, 9]
    m = Matrix.from_lists(I, J, V, 3, 3)
    n = Matrix.from_lists(I, J, W, 3, 3)

    o = m > n
    assert o.iseq(Matrix.from_lists(
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [False, False, False, False, False, False, False, False, False],
        ))

    o = m >= n
    assert o.iseq(Matrix.from_lists(
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [True, True, True, False, False, False, False, False, False],
        ))

    o = m < n
    assert o.iseq(Matrix.from_lists(
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [False, False, False, True, True, True, True, True, True],
        3, 3
        ))

    o = m <= n
    assert o.iseq(Matrix.from_lists(
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [True, True, True, True, True, True, True, True, True],
        3, 3
        ))

    o = m == n
    assert o.iseq(Matrix.from_lists(
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [True, True, True, False, False, False, False, False, False],
        3, 3
        ))

    o = m != n
    assert o.iseq(Matrix.from_lists(
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [False, False, False, True, True, True, True, True, True],
        3, 3
        ))

def test_select_cmp():
    I, J = tuple(map(list, zip(*product(range(3), repeat=2))))
    V = list(range(9))
    m = Matrix.from_lists(I, J, V, 3, 3)

    n = m.select('>', 5)
    assert n.iseq(Matrix.from_lists(
        [2, 2, 2], [0, 1, 2], [6, 7, 8]
        ))

    n = m.select('>=', 5)
    assert n.iseq(Matrix.from_lists(
        [1, 2, 2, 2],
        [2, 0, 1, 2],
        [5, 6, 7, 8]
        ))

    n = m.select('<', 5)
    assert n.iseq(Matrix.from_lists(
        [0, 0, 0, 1, 1],
        [0, 1, 2, 0, 1],
        [0, 1, 2, 3, 4],
        3, 3
        ))

    n = m.select('<=', 5)
    assert n.iseq(Matrix.from_lists(
        [0, 0, 0, 1, 1, 1],
        [0, 1, 2, 0, 1, 2],
        [0, 1, 2, 3, 4, 5],
        3, 3
        ))

    n = m.select('==', 5)
    assert n.iseq(Matrix.from_lists(
        [1],
        [2],
        [5],
        3, 3
        ))

    n = m.select('!=', 5)
    assert n.iseq(Matrix.from_lists(
        [0, 0, 0, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 0, 1, 2],
        [0, 1, 2, 3, 4, 6, 7, 8],
        3, 3
        ))

def test_shape_repr():
    m = Matrix.from_lists(
        [2, 2, 2], [0, 1, 2], [6, 7, 8]
    )
    assert m.shape == (3, 3)
    assert repr(m) == '<Matrix (3x3 : 3:INT64)>'

def test_dense():
    m = Matrix.dense(UINT8, 10, 10)
    assert len(m) == 100
    assert all(x[2] == 1 for x in m)
    m = Matrix.dense(UINT8, 10, 10, 0)
    assert len(m) == 100
    assert all(x[2] == 0 for x in m)

def test_identity():
    m = Matrix.identity(UINT8, 10)
    assert len(m) == 10
    for i in range(len(m)):
        assert m[i,i] == UINT8.aidentity

def test_to_arrays():
    m = Matrix.dense(UINT8, 10, 10)
    I, J, X = m.to_arrays()
    assert len(I) == 100
    assert len(J) == 100
    assert len(X) == 100
    assert all(x == 1 for x in X)

def test_pow():
    m = Matrix.dense(UINT8, 10, 10)
    assert m.identity(UINT8, 10) == m ** 0
    assert m == m ** 1
    assert (m @ m) == (m ** 2)
    vals = (m ** 3).to_arrays()[2]
    assert (x == 100 for x in vals)

def test_T():
    m = Matrix.dense(UINT8, 10, 10)
    assert m.T == m.transpose()
