import sys
from operator import mod
from itertools import product, repeat
import re

import pytest

from pygraphblas import *
from pygraphblas.base import lib, _check


def test_matrix_init_without_type():
    mx = Matrix.sparse(INT8)

    # get a raw Matrix pointer and wrap it without knowing its type
    new_mx = ffi.new("GrB_Matrix*")
    _check(lib.GrB_Matrix_dup(new_mx, mx._matrix[0]))
    mx2 = Matrix(new_mx)

    assert mx.type == mx2.type


def test_matrix_create():
    m = Matrix.sparse(INT8)
    assert m.nrows == lib.GxB_INDEX_MAX
    assert m.ncols == lib.GxB_INDEX_MAX
    assert m.nvals == 0
    m = Matrix.sparse(INT8, 10, 10)
    assert m.nrows == 10
    assert m.ncols == 10
    assert m.nvals == 0
    assert len(m) == 0
    m = Matrix.dense(INT8, 1, 1)
    assert m.nrows == 1
    assert m.ncols == 1
    assert m.nvals == 1
    with pytest.raises(AssertionError):
        m = Matrix.dense(INT8, 0, 0)
    m = Matrix.from_lists([0], [0], [0j])
    assert m.type == FC64
    assert m.nrows == 1
    assert m.ncols == 1
    assert m.nvals == 1


def test_matrix_get_set_element():
    m = Matrix.sparse(INT8, 10, 10)
    m[3, 3] = 3
    assert m.nrows == 10
    assert m.ncols == 10
    assert m.nvals == 1
    assert len(m) == 1
    assert m[3, 3] == 3


def test_matrix_slice_vector():
    v = Matrix.from_lists(list(range(10)), list(range(10)), list(range(10)))
    assert v[5] == Vector.from_lists([5], [5], 10)


def test_clear():
    v = Matrix.from_lists(list(range(10)), list(range(10)), list(range(10)))
    assert v.nvals == 10
    assert len(v) == 10
    v.clear()
    assert v.nvals == 0
    assert len(v) == 0


def test_resize():
    v = Matrix.from_lists(list(range(10)), list(range(10)), list(range(10)))
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
    m = Matrix.sparse(INT8, 10, 10)
    m[3, 3] = 3
    n = Matrix.dup(m)
    assert n.nrows == 10
    assert n.ncols == 10
    assert n.nvals == 1
    assert n[3, 3] == 3


def test_matrix_to_from_lists():
    v = Matrix.from_lists(list(range(10)), list(range(10)), list(range(10)))
    assert v.nrows == 10
    assert v.ncols == 10
    assert v.nvals == 10
    assert v.to_lists() == [
        list(range(10)),
        list(range(10)),
        list(range(10)),
    ]


def test_matrix_gb_type():
    v = Matrix.sparse(BOOL, 10)
    assert v.gb_type == lib.GrB_BOOL
    v = Matrix.sparse(INT8, 10)
    assert v.gb_type == lib.GrB_INT8
    v = Matrix.sparse(FP64, 10)
    assert v.gb_type == lib.GrB_FP64


def test_matrix_eadd():
    I = list(range(10))
    v = Matrix.from_lists(I, I, I)
    v[0, 1] = 1
    w = Matrix.from_lists(I, I, I)
    w[1, 0] = 1

    addition_ref = Matrix.from_lists(I, I, list(range(0, 20, 2)))
    addition_ref[0, 1] = 1
    addition_ref[1, 0] = 1

    sum1 = v.eadd(w)
    assert sum1.iseq(addition_ref)
    sum1 = v + w
    assert sum1.iseq(addition_ref)
    sum1 = v.eadd(w, v.type.SECOND)
    sum2 = v | w
    assert sum1.iseq(sum2)
    sum3 = v.dup()
    sum3 |= w
    assert sum3.iseq(sum2)

    prod_ref = Matrix.from_lists(I, I, [0, 1, 4, 9, 16, 25, 36, 49, 64, 81])
    prod_ref[0, 1] = 1
    prod_ref[1, 0] = 1

    prod5 = v.eadd(w, "*")
    assert prod5.iseq(prod_ref)
    assert prod5.isne(sum1)


def test_sub():
    I = list(range(10))
    v = Matrix.from_lists(I, I, I)
    v[0, 1] = 1
    w = Matrix.from_lists(I, I, I)
    w[1, 0] = 1
    # subtraction (explicit zeros, if same numbers are subtracted)
    subtraction_ref = Matrix.from_lists(I, I, [0] * 10)
    # 1 - empty = 1
    subtraction_ref[0, 1] = 1
    # empty - 1 = -1 (assuming implicit zero for elements not present)
    subtraction_ref[1, 0] = 1

    diff1 = v - w
    assert diff1.iseq(subtraction_ref)
    diff2 = v.dup()
    diff2 -= w
    assert diff2.iseq(subtraction_ref)


def test_matrix_emult():
    I = list(range(10))
    V = list(range(1, 10 + 1))
    v = Matrix.from_lists(I, I, V)
    w = Matrix.from_lists(I, I, V)

    mult1 = v.emult(w)
    assert mult1.iseq(Matrix.from_lists(I, I, [v * v for v in V]))
    mult1 = v.emult(w, v.type.SECOND)
    mult2 = v & w
    assert mult2.iseq(mult1)
    mult3 = v.dup()
    mult3 &= w
    assert mult3.iseq(mult2)

    # division
    division_ref = Matrix.from_lists(I, I, [1] * 10)
    div1 = v / w
    assert div1.iseq(division_ref)
    div2 = v.dup()
    div2 /= w
    assert div2.iseq(division_ref)


def test_matrix_reduce_bool():
    v = Matrix.sparse(BOOL, 10, 10)
    assert not v.reduce_bool()
    v[3, 3] = True
    v[4, 4] = False
    assert v.reduce_bool() == True
    with BOOL.LAND_MONOID:
        assert v.reduce_bool() == False


def test_matrix_reduce_int():
    v = Matrix.sparse(INT8, 10, 10)
    r = v.reduce_int()
    assert type(r) is int
    assert r == 0
    v[3, 3] = 3
    v[4, 4] = 4
    assert v.reduce_int() == 7
    with INT8.TIMES_MONOID:
        assert v.reduce_int() == 12


def test_matrix_reduce_float():
    v = Matrix.sparse(FP64, 10, 10)
    r = v.reduce_float()
    assert type(r) is float
    assert r == 0.0
    v[3, 3] = 3.3
    v[4, 4] = 4.4
    assert v.reduce_float() == 7.7
    with FP64.TIMES_MONOID:
        assert v.reduce_float() == 14.52
    assert v.reduce_float(FP64.TIMES_MONOID) == 14.52


def test_matrix_reduce_vector():
    m = Matrix.from_lists(list(range(10)), list(range(10)), list(range(10)))
    v = m.reduce_vector()
    v == Vector.from_list(list(range(10)))


def test_mxm():
    m = Matrix.from_lists([0, 1, 2], [1, 2, 0], [1, 2, 3])
    n = Matrix.from_lists([0, 1, 2], [1, 2, 0], [2, 3, 4])
    o = m.mxm(n)
    assert o.nrows == 3
    assert o.ncols == 3
    assert o.nvals == 3
    r = Matrix.from_lists([0, 1, 2], [2, 0, 1], [3, 8, 6])
    assert o.iseq(r)
    assert r.iseq(m @ n)
    m @= n
    assert r.iseq(m)
    o = m.mxm(n, semiring=BOOL.LOR_LAND)
    assert o.iseq(Matrix.from_lists([0, 1, 2], [0, 1, 2], [True, True, True]))


def test_mxm_context():
    m = Matrix.from_lists([0, 1, 2], [1, 2, 0], [1, 2, 3])
    n = Matrix.from_lists([0, 1, 2], [1, 2, 0], [2, 3, 4])

    with INT64.PLUS_PLUS:
        o = m @ n

    assert o.iseq(Matrix.from_lists([0, 1, 2], [2, 0, 1], [4, 6, 5]))

    with BOOL.LOR_LAND:
        o = m @ n
    assert o.iseq(Matrix.from_lists([0, 1, 2], [2, 0, 1], [True, True, True]))

    with descriptor.T0:
        o = m @ n

    assert o.iseq(m.mxm(n, desc=descriptor.T0))

    with BOOL.LOR_LAND:
        o = m @ n
    assert o.iseq(Matrix.from_lists([0, 1, 2], [2, 0, 1], [True, True, True]))

    with pytest.raises(TypeError):
        m @ 3
    with pytest.raises(TypeError):
        m @ Scalar.from_value(3)


def test_mxv():
    m = Matrix.from_lists([0, 1, 2, 3], [1, 2, 0, 1], [1, 2, 3, 4])
    v = Vector.from_lists([0, 1, 2], [2, 3, 4])
    o = m.mxv(v)
    assert o.iseq(Vector.from_lists([0, 1, 2, 3], [3, 8, 6, 12]))

    assert o.iseq(m @ v)

    assert o.iseq(m.transpose().mxv(v, desc=descriptor.T0))

    with INT64.PLUS_PLUS:
        o = m.mxv(v)
        assert o.iseq(Vector.from_lists([0, 1, 2, 3], [4, 6, 5, 7]))
        assert o.iseq(m @ v)


def test_matrix_pattern():
    v = Matrix.from_lists(list(range(10)), list(range(10)), list(range(10)))
    p = v.pattern()
    assert p.gb_type == lib.GrB_BOOL
    assert p.nrows == 10
    assert p.ncols == 10
    assert p.nvals == 10


def test_matrix_transpose():
    v = Matrix.from_lists(
        list(range(2, -1, -1)), list(range(3)), list(range(3)), nrows=3, ncols=4
    )
    w = v.transpose()
    assert w.iseq(Matrix.from_lists([0, 1, 2], [2, 1, 0], [0, 1, 2], nrows=4, ncols=3))
    v2 = v.transpose(desc=descriptor.T0)
    assert v2.iseq(v)


def test_matrix_mm_read_write(tmp_path):
    mmf = tmp_path / "mmwrite_test.mm"
    mmf.touch()
    m = Matrix.from_lists([0, 1, 2], [0, 1, 2], [2, 3, 4])
    with mmf.open("w") as f:
        m.to_mm(f)
    with mmf.open() as f:
        assert f.readlines() == [
            "%%MatrixMarket matrix coordinate integer symmetric\n",
            "%%GraphBLAS GrB_INT64\n",
            "3 3 3\n",
            "1 1 2\n",
            "2 2 3\n",
            "3 3 4\n",
        ]

    with mmf.open() as f:
        n = Matrix.from_mm(f, INT8)
    assert n.iseq(m)


def test_matrix_binfile_read_write(tmp_path):
    binfilef = tmp_path / "binfilewrite_test.binfile"
    binfilef.touch()
    m = Matrix.from_lists([0, 1, 2], [0, 1, 2], [2, 3, 4])
    bbf = bytes(binfilef)
    m.to_binfile(bbf)
    n = Matrix.from_binfile(bbf)
    assert n.iseq(m)


def test_matrix_tsv_read(tmp_path):
    mmf = tmp_path / "tsv_test.mm"
    mmf.touch()
    with mmf.open("w") as f:
        f.writelines(["3\t3\t3\n", "1\t1\t2\n", "2\t2\t3\n", "3\t3\t4\n"])

    with mmf.open() as f:
        n = Matrix.from_tsv(f, INT8, 3, 3)
    assert n.to_lists() == [[0, 1, 2], [0, 1, 2], [2, 3, 4]]


def test_matrix_random():
    m = Matrix.random(INT8, 10, 10, 5, seed=42)
    assert m.nrows == 10
    assert m.ncols == 10
    assert len(list(m)) == 5
    m = Matrix.random(INT8, 10, 10, 5)
    assert m.nrows == 10
    assert m.ncols == 10
    assert 0 < len(list(m)) <= 5


def test_matrix_slicing():
    I, J = tuple(map(list, zip(*product(range(3), repeat=2))))
    V = list(range(9))
    m = Matrix.from_lists(I, J, V, 3, 3)
    # slice out row vector
    v = m[2]
    assert v == Vector.from_lists([0, 1, 2], [6, 7, 8])
    # slice out row vector from rectangular matrix
    v = m[:, 0:1][2]
    assert v == Vector.from_lists([0, 1], [6, 7])

    # slice out row vector
    v = m[2, :]
    assert v == Vector.from_lists([0, 1, 2], [6, 7, 8])

    # slice out column vector
    v = m[:, 2]
    assert v == Vector.from_lists([0, 1, 2], [2, 5, 8])

    # slice copy
    n = m[:]
    assert n.iseq(m)
    # also slice copy
    n = m[:, :]
    assert n.iseq(m)

    # submatrix slice out rows
    sm = m[0:1]
    assert sm.iseq(
        Matrix.from_lists(
            [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2], [0, 1, 2, 3, 4, 5], 2, 3
        )
    )

    # submatrix slice out columns
    n = m[:, 1:]
    assert n.iseq(
        Matrix.from_lists(
            [0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1], [1, 2, 4, 5, 7, 8], 3, 2
        )
    )

    # submatrix slice out column range
    sm = m[:, 1:2]
    assert sm.iseq(
        Matrix.from_lists(
            [0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1], [1, 2, 4, 5, 7, 8], 3, 2
        )
    )

    # submatrix slice out listed columns
    sm = m[:, [0, 2]]
    assert sm.iseq(
        Matrix.from_lists(
            [0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1], [0, 2, 3, 5, 6, 8], 3, 2
        )
    )

    # submatrix slice out rows
    n = m[1:, :]
    assert n.iseq(
        Matrix.from_lists(
            [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2], [3, 4, 5, 6, 7, 8], 2, 3
        )
    )

    # submatrix slice out row range
    sm = m[1:2, :]
    assert sm.iseq(
        Matrix.from_lists(
            [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2], [3, 4, 5, 6, 7, 8], 2, 3
        )
    )

    # submatrix slice out listed rows
    sm = m[[0, 2], :]
    assert sm.iseq(
        Matrix.from_lists(
            [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2], [0, 1, 2, 6, 7, 8], 2, 3
        )
    )


def test_matrix_assign():
    m = Matrix.from_lists(list(range(3)), list(range(3)), list(range(3)))
    assert m.nvals == 3

    m[2] = Vector.from_list(list(repeat(6, 3)))
    assert m.nvals == 5
    assert m.iseq(
        Matrix.from_lists([0, 1, 2, 2, 2], [0, 1, 0, 1, 2], [0, 1, 6, 6, 6], 3, 3)
    )

    m = Matrix.from_lists(list(range(3)), list(range(3)), list(range(3)))
    assert m.nvals == 3

    m[2, :] = Vector.from_list(list(repeat(6, 3)))
    assert m.nvals == 5
    assert m.iseq(
        Matrix.from_lists([0, 1, 2, 2, 2], [0, 1, 0, 1, 2], [0, 1, 6, 6, 6], 3, 3)
    )

    m = Matrix.from_lists(list(range(3)), list(range(3)), list(range(3)))

    assert m.nvals == 3
    m[:, 2] = Vector.from_list(list(repeat(6, 3)))
    assert m.nvals == 5
    assert m.iseq(
        Matrix.from_lists([0, 1, 0, 1, 2], [0, 1, 2, 2, 2], [0, 1, 6, 6, 6], 3, 3)
    )

    m = Matrix.sparse(INT64, 3, 3)
    assert m.nvals == 0
    n = Matrix.from_lists([0, 1, 2], [0, 1, 2], [0, 1, 2])
    m[:, :] = n
    assert m.iseq(n)

    n = Matrix.from_lists([0, 1, 2], [0, 1, 2], [0, 1, 2])

    mask = m > 0
    m[mask] = 9
    assert m.iseq(Matrix.from_lists([0, 1, 2], [0, 1, 2], [0, 9, 9]))
    with pytest.raises(TypeError):
        m[mask] = ""

    m = Matrix.sparse(INT64, 3, 3)
    assert m.nvals == 0
    m[:] = 4
    assert m.iseq(Matrix.dense(INT64, 3, 3, 4))

    m = Matrix.sparse(INT64, 3, 3)
    n = Matrix.from_lists([0, 1, 2], [0, 1, 2], [2, 2, 2])
    assert m.nvals == 0
    m[:] = n
    assert m.iseq(Matrix.identity(INT64, 3, 2))
    with pytest.raises(TypeError):
        m[""] = n


def test_kronecker():
    n = Matrix.from_lists(list(range(3)), list(range(3)), list(range(3)))
    m = Matrix.from_lists(list(range(3)), list(range(3)), list(range(3)))

    o = n.kronecker(m)
    assert o.iseq(
        Matrix.from_lists(
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 0, 0, 0, 1, 2, 0, 2, 4],
        )
    )


def test_apply():
    v = Matrix.from_lists([0, 1, 2], [0, 1, 2], [2, 3, 4])
    w = v.apply(INT64.AINV)
    assert w.iseq(Matrix.from_lists([0, 1, 2], [0, 1, 2], [-2, -3, -4]))

    w2 = v.apply(INT64.AINV)
    assert w.iseq(w2)

    w3 = v.apply(INT64.AINV)
    assert w.iseq(w3)


def test_get_set_options():
    v = Matrix.random(INT8, 10, 10, 10, seed=42)
    v.hyper_switch = lib.GxB_ALWAYS_HYPER
    v.format = lib.GxB_BY_COL
    assert v.hyper_switch == lib.GxB_ALWAYS_HYPER
    assert v.format == lib.GxB_BY_COL
    assert v.sparsity_control == lib.GxB_AUTO_SPARSITY
    assert v.sparsity_status == lib.GxB_HYPERSPARSE

    v.hyper_switch = 2.0
    assert v.hyper_switch == 2.0

    v.format = lib.GxB_BY_ROW
    assert v.format == lib.GxB_BY_ROW

    v.sparsity_control = lib.GxB_BITMAP + lib.GxB_FULL
    assert v.sparsity_control == lib.GxB_BITMAP + lib.GxB_FULL

    w = Matrix.sparse(INT8, 10, 10)
    assert w.hyper_switch == lib.GxB_HYPER_DEFAULT
    assert w.format == lib.GxB_BY_ROW
    assert w.sparsity_control == lib.GxB_AUTO_SPARSITY
    assert w.sparsity_status == lib.GxB_HYPERSPARSE


def test_square():
    v = Matrix.random(INT8, 10, 10, 10, seed=42)
    assert v.square
    w = Matrix.random(INT8, 10, 9, 10, seed=42)
    assert not w.square


def test_select():
    v = Matrix.from_lists([0, 1, 2], [0, 1, 2], [0, 0, 3])
    w = v.select(lib.GxB_NONZERO)
    assert w.to_lists() == [[2], [2], [3]]

    w = v.select("!=0")
    assert w.to_lists() == [[2], [2], [3]]

    w = v.select("!=", 0)
    assert w.to_lists() == [[2], [2], [3]]

    w = v.select(">", 0)
    assert w.to_lists() == [[2], [2], [3]]

    w = v.select("<", 3)
    assert w.to_lists() == [[0, 1], [0, 1], [0, 0]]

    w = v.select(">=", 0)
    assert w.iseq(v)

    w = v.select(">=0")
    assert w.iseq(v)

    # with unaryop.NONZERO:
    #     w = v.select()
    # assert w.to_lists() == [[2], [2], [3]]


def test_select_ops():
    I, J = tuple(map(list, zip(*product(range(3), repeat=2))))
    V = list(range(9))
    m = Matrix.from_lists(I, J, V, 3, 3)

    assert m.tril().iseq(
        Matrix.from_lists([0, 1, 1, 2, 2, 2], [0, 0, 1, 0, 1, 2], [0, 3, 4, 6, 7, 8])
    )

    assert m.triu().iseq(
        Matrix.from_lists([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2], [0, 1, 2, 4, 5, 8])
    )

    assert m.diag().iseq(Matrix.from_lists([0, 1, 2], [0, 1, 2], [0, 4, 8]))

    assert m.offdiag().iseq(
        Matrix.from_lists([0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1], [1, 2, 3, 5, 6, 7])
    )

    assert m.nonzero().iseq(
        Matrix.from_lists(
            [0, 0, 1, 1, 1, 2, 2, 2], [1, 2, 0, 1, 2, 0, 1, 2], [1, 2, 3, 4, 5, 6, 7, 8]
        )
    )

    assert (-m).iseq(
        Matrix.from_lists(
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, -1, -2, -3, -4, -5, -6, -7, -8],
        )
    )

    n = -m

    assert abs(m).iseq(
        Matrix.from_lists(
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
        )
    )

    m = Matrix.from_lists([0, 1, 2], [0, 1, 2], [0.0, 1.0, 2.0], 3, 3)

    n = ~m
    assert n.iseq(Matrix.from_lists([0, 1, 2], [0, 1, 2], [float("inf"), 1.0, 0.5]))


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
        [0, 0, 0, 1, 1], [0, 1, 2, 0, 1], [True, True, True, True, True], 3, 3
    )

    n = m <= 5
    assert n == Matrix.from_lists(
        [0, 0, 0, 1, 1, 1],
        [0, 1, 2, 0, 1, 2],
        [True, True, True, True, True, True],
        3,
        3,
    )

    n = m == 5
    assert n == Matrix.from_lists([1], [2], [5], 3, 3)

    n = m != 5
    assert n == Matrix.from_lists(
        [0, 0, 0, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 0, 1, 2],
        [0, 1, 2, 3, 4, 6, 7, 8],
        3,
        3,
    )
    with pytest.raises(TypeError):
        m < ""


def test_cmp():
    I, J = tuple(map(list, zip(*product(range(3), repeat=2))))
    V = list(range(9))
    W = [0, 1, 2, 4, 5, 6, 7, 8, 9]
    m = Matrix.from_lists(I, J, V, 3, 3)
    n = Matrix.from_lists(I, J, W, 3, 3)

    o = m > n
    assert o.iseq(
        Matrix.from_lists(
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [False, False, False, False, False, False, False, False, False],
        )
    )

    o = m >= n
    assert o.iseq(
        Matrix.from_lists(
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [True, True, True, False, False, False, False, False, False],
        )
    )

    o = m < n
    assert o.iseq(
        Matrix.from_lists(
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [False, False, False, True, True, True, True, True, True],
            3,
            3,
        )
    )

    o = m <= n
    assert o.iseq(
        Matrix.from_lists(
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [True, True, True, True, True, True, True, True, True],
            3,
            3,
        )
    )

    o = m == n
    assert o.iseq(
        Matrix.from_lists(
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [True, True, True, False, False, False, False, False, False],
            3,
            3,
        )
    )

    o = m != n
    assert o.iseq(
        Matrix.from_lists(
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [False, False, False, True, True, True, True, True, True],
            3,
            3,
        )
    )


def test_select_cmp():
    I, J = tuple(map(list, zip(*product(range(3), repeat=2))))
    V = list(range(9))
    m = Matrix.from_lists(I, J, V, 3, 3)

    n = m.select(">", 5)
    assert n.iseq(Matrix.from_lists([2, 2, 2], [0, 1, 2], [6, 7, 8]))

    n = m.select(">=", 5)
    assert n.iseq(Matrix.from_lists([1, 2, 2, 2], [2, 0, 1, 2], [5, 6, 7, 8]))

    n = m.select("<", 5)
    assert n.iseq(
        Matrix.from_lists([0, 0, 0, 1, 1], [0, 1, 2, 0, 1], [0, 1, 2, 3, 4], 3, 3)
    )

    n = m.select("<=", 5)
    assert n.iseq(
        Matrix.from_lists(
            [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2], [0, 1, 2, 3, 4, 5], 3, 3
        )
    )

    n = m.select("==", 5)
    assert n.iseq(Matrix.from_lists([1], [2], [5], 3, 3))

    n = m.select("!=", 5)
    assert n.iseq(
        Matrix.from_lists(
            [0, 0, 0, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 0, 1, 2],
            [0, 1, 2, 3, 4, 6, 7, 8],
            3,
            3,
        )
    )


def test_shape_repr():
    m = Matrix.from_lists([2, 2, 2], [0, 1, 2], [6, 7, 8])
    assert m.shape == (3, 3)
    assert repr(m) == "<Matrix (3x3 : 3:INT64)>"


def test_iters():
    m = Matrix.from_lists([2, 2, 2], [0, 1, 2], [6, 7, 8])
    assert m.shape == (3, 3)
    assert list(m) == [(2, 0, 6), (2, 1, 7), (2, 2, 8)]
    assert [(i, j, k) for i, j, k in m] == [(2, 0, 6), (2, 1, 7), (2, 2, 8)]
    assert list(m.rows) == [2, 2, 2]
    assert list(m.cols) == [0, 1, 2]
    assert list(m.vals) == [6, 7, 8]


def test_dense():
    m = Matrix.dense(UINT8, 10, 10)
    assert len(m) == 100
    assert all(x[2] == 0 for x in m)
    m = Matrix.dense(UINT8, 10, 10, 1)
    assert len(m) == 100
    assert all(x[2] == 1 for x in m)


def test_identity():
    m = Matrix.identity(UINT8, 10)
    assert len(m) == 10
    for i in range(len(m)):
        assert m[i, i] == UINT8.one


def test_to_arrays():
    m = Matrix.dense(UINT8, 10, 10)
    I, J, X = m.to_arrays()
    assert len(I) == 100
    assert len(J) == 100
    assert len(X) == 100
    assert all(x == 0 for x in X)

    m = Matrix.dense(FC64, 10, 10)
    with pytest.raises(TypeError):
        I, J, X = m.to_arrays()


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


def test_to_string():
    M = Matrix.from_lists(*(map(list, zip((0, 1, 10), (1, 0, 11)))))
    assert re.search("-.*10.*\n.*11.*-", M.to_string(empty_char="-"))


# def test_complex():
#     m = Matrix.sparse(Complex, 10, 10)
#     m[2,3] = 3+4j
#     assert m[2,3] == 3+4j


def test_get_contains():
    m = Matrix.identity(UINT8, 10)
    for i in range(m.nrows):
        for j in range(m.ncols):
            if i == j:
                assert (i, j) in m
                assert m.get(i, j) == 1
            else:
                assert (i, j) not in m
                assert m.get(i, j) is None


def scalar_assign():
    m = Matrix.sparse(UINT8, 10, 10)
    m.assign_scalar(42, 1, 1)
    assert m[1, 1] == 42
    m.assign_scalar(43, 2, 2)
    assert m[2, 2] == 43


def test_wait():
    m = Matrix.sparse(UINT8, 10, 10)
    m[:, :] = 1
    m.wait()


def test_apply_first():
    m = Matrix.from_lists([0, 1], [0, 1], [4, 2])
    assert m.apply_first(2, INT8.PLUS).to_lists() == [[0, 1], [0, 1], [6, 4]]


def test_apply_second():
    m = Matrix.from_lists([0, 1], [0, 1], [5, 1])
    assert m.apply_second(INT8.MINUS, 2).to_lists() == [[0, 1], [0, 1], [3, -1]]


def test_add_scalar():
    m = Matrix.from_lists([0, 1], [0, 1], [5, 1])
    assert (m + 3).to_lists() == [[0, 1], [0, 1], [8, 4]]


def test_radd_scalar():
    m = Matrix.from_lists([0, 1], [0, 1], [5, 1])
    assert (3 + m).to_lists() == [[0, 1], [0, 1], [8, 4]]


def test_iadd_scalar():
    m = Matrix.from_lists([0, 1], [0, 1], [5, 1])
    m += 3
    assert m.to_lists() == [[0, 1], [0, 1], [8, 4]]


def test_sub_scalar():
    m = Matrix.from_lists([0, 1], [0, 1], [5, 1])
    assert (m - 3).to_lists() == [[0, 1], [0, 1], [2, -2]]


def test_rsub_scalar_second():
    m = Matrix.from_lists([0, 1], [0, 1], [5, 1])
    assert (3 - m).to_lists() == [[0, 1], [0, 1], [-2, 2]]


def test_isub_scalar():
    m = Matrix.from_lists([0, 1], [0, 1], [5, 1])
    m -= 3
    assert m.to_lists() == [[0, 1], [0, 1], [2, -2]]


def test_mul_scalar():
    m = Matrix.from_lists([0, 1], [0, 1], [5, 1])
    assert (m * 3).to_lists() == [[0, 1], [0, 1], [15, 3]]


def test_rmul_scalar_second():
    m = Matrix.from_lists([0, 1], [0, 1], [5, 1])
    assert (3 * m).to_lists() == [[0, 1], [0, 1], [15, 3]]


def test_imul_scalar():
    m = Matrix.from_lists([0, 1], [0, 1], [5, 1])
    m *= 3
    assert m.to_lists() == [[0, 1], [0, 1], [15, 3]]


def test_truediv_scalar():
    m = Matrix.from_lists([0, 1], [0, 1], [15, 3])
    assert (m / 3).to_lists() == [[0, 1], [0, 1], [5, 1]]


def test_rtruediv_scalar_second():
    m = Matrix.from_lists([0, 1], [0, 1], [3, 5])
    assert (15 / m).to_lists() == [[0, 1], [0, 1], [5, 3]]


def test_itruediv_scalar():
    m = Matrix.from_lists([0, 1], [0, 1], [15, 3])
    m /= 3
    assert m.to_lists() == [[0, 1], [0, 1], [5, 1]]


def test_delitem():
    m = Matrix.from_lists([0, 1], [0, 1], [4, 2])
    assert len(m) == 2
    del m[0, 0]
    assert len(m) == 1
    assert m[1, 1] == 2
    with pytest.raises(TypeError):
        del m[""]
    with pytest.raises(TypeError):
        del m["", 0]
    with pytest.raises(TypeError):
        del m[0, ""]


def test_cast():
    m = Matrix.from_lists([0, 1], [0, 1], [4, 2])
    n = m.cast(FP64)
    assert n.iseq(Matrix.from_lists([0, 1], [0, 1], [4.0, 2.0]))


def test_promotion():
    m = Matrix.from_lists([0, 1], [0, 1], [4, 2], typ=FP32)
    n = Matrix.from_lists([0, 1], [0, 1], [4, 2], typ=FP64)
    o = m @ n
    assert o.type == FP64
    n = Matrix.from_lists([0, 1], [0, 1], [4, 2], typ=UINT8)
    o = m @ n
    assert o.type == FP32

    m = Matrix.from_lists([0, 1], [0, 1], [-4, 2], typ=INT8)
    o = m @ n
    assert o.type == INT8


def test_str():
    m = Matrix.from_lists([0, 1], [0, 1], [4, 2], typ=INT8)
    assert (
        str(m)
        == """\
      0  1
  0|  4   |  0
  1|     2|  1
      0  1"""
    )

    b = Matrix.from_lists([0, 1], [0, 1], [True, True])
    assert (
        str(b)
        == """\
      0  1
  0|  t   |  0
  1|     t|  1
      0  1"""
    )


def test_nonzero():
    m = Matrix.sparse(INT8, 10, 10)
    assert not bool(m)
    m = Matrix.from_lists(list(range(3)), list(range(3)), list(range(3)))
    assert bool(m)


def test_to_scipy_sparse():
    v = Matrix.random(INT8, 10, 10, 4, seed=42)
    assert len(v) == 4
    s = v.to_scipy_sparse()
    assert (s.data == [63, 105, 17, 20]).all()
    m = v.to_numpy()
    assert m.shape == (10, 10)
