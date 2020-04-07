import sys
from itertools import repeat
from array import array
import re

from pygraphblas import *

def test_vector_create_from_type():
    m = Vector.from_type(INT64)
    assert m.size == 0
    assert m.nvals == 0
    m = Vector.from_type(INT64, 10)
    assert m.size == 10

def test_vector_gb_type():
    v = Vector.from_type(BOOL, 10)
    assert v.gb_type == lib.GrB_BOOL
    v = Vector.from_type(INT64, 10)
    assert v.gb_type == lib.GrB_INT64
    v = Vector.from_type(FP64, 10)
    assert v.gb_type == lib.GrB_FP64

def test_vector_set_element():
    m = Vector.from_type(INT64, 10)
    m[3] = 3
    assert m.size == 10
    assert m.nvals == 1
    assert len(list(m)) == 1
    assert m[3] == 3
    m = Vector.from_type(BOOL, 10)
    m[3] = True
    assert m.size == 10
    assert m.nvals == 1
    assert m[3] == True
    m = Vector.from_type(FP64, 10)
    m[3] = 3.3
    assert m.size == 10
    assert m.nvals == 1
    assert m[3] == 3.3

def test_vector_create_dup():
    m = Vector.from_type(INT64, 10)
    m[3] = 3
    n = Vector.dup(m)
    assert n.size == 10
    assert n.nvals == 1
    assert n[3] == 3

def test_vector_from_list():
    v = Vector.from_list(list(range(10)))
    assert v.size == 10
    assert v.nvals == 10
    for i in range(10):
        assert i == v[i]

def test_vector_to_lists():
    v = Vector.from_list(list(range(10)))
    assert v.to_lists() == [
        list(range(10)),
        list(range(10)),
        ]

def test_vector_eq():
    v = Vector.from_list(list(range(10)))
    w = Vector.from_list(list(range(10)))
    x = Vector.from_list(list(range(1,11)))
    assert v.iseq(w)
    assert v.isne(x)

def test_vector_eadd():
    V = list(range(2, 10))
    v = Vector.from_lists(V, V)
    v[0] = 1
    w = Vector.from_lists(V, V)
    w[1] = 1

    addition_ref = Vector.from_lists(V, list(range(2*2, 2*10, 2)))
    addition_ref[0] = 1
    addition_ref[1] = 1

    sum1 = v.eadd(w)
    assert sum1.iseq(addition_ref)
    sum2 = v + w
    assert sum1.iseq(sum2)
    sum3 = v.dup()
    sum3 += w
    assert sum3.iseq(sum2)

    # subtraction:
    # 1 - empty = 1
    # empty - 1 = -1 (assuming implicit 0)
    # explicit zeros where same numbers are subtracted
    subtraction_ref = Vector.from_list([1, -1] + ([0] * 8))

    diff1 = v - w
    assert diff1.iseq(subtraction_ref)
    diff2 = v.dup()
    diff2 -= w
    assert diff2.iseq(subtraction_ref)

def test_vector_emult():
    V = list(range(1, 10 + 1))
    v = Vector.from_list(V)
    w = Vector.from_list(V)
    mul1 = v.emult(w)
    assert mul1.iseq(Vector.from_list([v * v for v in V]))
    mul2 = v * w
    assert mul1.iseq(mul2)
    mul3 = v.dup()
    mul3 *= w
    assert mul3.iseq(mul2)

    # division
    division_ref = Vector.from_list([1] * 10)
    div1 = v / w
    assert div1.iseq(division_ref)
    div2 = v.dup()
    div2 /= w
    assert div2.iseq(division_ref)

def test_vector_reduce_bool():
    v = Vector.from_type(BOOL, 10)
    assert not v.reduce_bool()
    v[3] = True
    assert v.reduce_bool()

def test_vector_reduce_int():
    v = Vector.from_type(INT64, 10)
    r = v.reduce_int()
    assert type(r) is int
    assert r == 0
    v[3] = 3
    v[4] = 4
    assert v.reduce_int() == 7

def test_vector_reduce_float():
    v = Vector.from_type(FP64, 10)
    r = v.reduce_float()
    assert type(r) is float
    assert r == 0.0
    v[3] = 3.3
    v[4] = 4.4
    assert v.reduce_float() == 7.7

def test_vector_slice():
    v = Vector.from_list(list(range(10)))
    w = v[:9]
    assert w.size == 10
    assert w.nvals == 10
    assert w.to_lists() == [
        list(range(10)),
        list(range(10))]
    w = v[1:8]
    assert w.size == 8
    assert w.nvals == 8
    assert w.to_lists() == [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7, 8]]
    w = v[1:]
    assert w.size == 9
    assert w.nvals == 9
    assert w.to_lists() == [
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8, 9]]
    w = v[1:9:2]
    assert w.size == 5
    assert w.nvals == 5
    assert w.to_lists() == [
        [0, 1, 2, 3, 4],
        [1, 3, 5, 7, 9]]
    w = v[7:1:-2]
    assert w.size == 4
    assert w.nvals == 4
    assert w.to_lists() == [
        [0, 1, 2, 3],
        [7, 5, 3, 1]]
    # slice vector based on listed indices
    indices = [2, 3, 5, 7]
    w = v[indices]
    assert w.size == len(indices)
    assert w.nvals == len(indices)
    assert w.to_lists() == [
        list(range(len(indices))),
        indices]

def test_vector_assign():
    v = Vector.from_type(INT64, 10)
    assert v.nvals == 0
    w = Vector.from_lists(
        list(range(10)),
        list(range(10)))
    v[:] = w
    assert v.iseq(w)

    v[1:] = w[9:1:-1]
    assert v == Vector.from_lists(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 9, 8, 7, 6, 5, 4, 3, 2, 1])

    w[9:1:-1] = v[9:1:-1]
    assert w.iseq(v)

    v[:] = 3
    assert v.iseq(Vector.from_lists(
        list(range(10)),
        list(repeat(3, 10))))

    v[1:] = 0
    assert v.iseq(Vector.from_lists(
        list(range(10)),
        [3] + list(repeat(0, 9))))

def test_vxm():
    m = Matrix.from_lists(
        [0, 1, 2],
        [1, 2, 0],
        [1, 2, 3])
    v = Vector.from_lists(
        [0, 1, 2],
        [2, 3, 4])
    o = v.vxm(m)
    assert o.iseq(Vector.from_lists(
        [0, 1, 2],
        [12, 2, 6]))

    w = Vector.dup(v)

    assert (v @ m).iseq(o)
    # v @= m
    # assert v.iseq(o)

def test_apply():
    v = Vector.from_lists(
        [0, 1, 2],
        [2.0, 4.0, 8.0])

    w = v.apply(INT64.AINV)
    assert w.iseq(Vector.from_lists(
        [0, 1, 2],
        [-2.0, -4.0, -8.0]))

    w = ~v
    assert w.iseq(Vector.from_lists(
        [0, 1, 2],
        [0.5, 0.25, 0.125]))

def test_select():
    v = Vector.from_lists(
        [0, 1, 2],
        [0, 0, 3])
    w = v.select(lib.GxB_NONZERO)
    assert w.to_lists() == [[2], [3]]

def test_to_dense():
    v = Vector.from_lists(list(range(0, 6, 2)), list(range(3)))
    assert v.size == 5
    assert v.nvals == 3
    w = v.to_dense()
    assert w.nvals == 5
    assert w.iseq(Vector.from_lists(
        [0, 1, 2, 3, 4],
        [0, 0, 1, 0, 2]))

def test_dense():
    m = Vector.dense(UINT8, 10)
    assert len(m) == 10
    assert all(x[1] == 0 for x in m)
    m = Vector.dense(UINT8, 10, 1)
    assert len(m) == 10
    assert all(x[1] == 1 for x in m)

def test_compare():
    v = Vector.from_lists(
        [0, 1, 2],
        [0, 1, 3])
    assert (v > 2).iseq(Vector.from_lists(
        [0, 1, 2], [False, False, True]))
    assert (v < 2).iseq(Vector.from_lists(
        [0, 1], [True, True], 3))

def test_1_to_n():
    v = Vector.from_1_to_n(10)
    assert v.iseq(Vector.from_lists(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        typ=types.INT32
        ))
    # this takes too much ram
    # w = Vector.from_1_to_n(lib.INT32_MAX + 1)
    # assert w.type == lib.INT64

def test_to_arrays():
    v = Vector.from_1_to_n(10)
    assert v.to_arrays() == (
        array('L', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        array('l', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

def test_contains():
    v = Vector.from_lists(
        [0, 1, 9, 20],
        [0, 1, 3, 4])
    assert 1 in v
    assert 9 in v
    assert 10 not in v

def test_to_string():
    v = Vector.from_lists(*(map(list, zip((0, 10), (2, 11)))))
    assert re.search('10.*\n.*-.*\n.*11', v.to_string(empty_char='-'))

def test_get_contains():
    v = Vector.dense(UINT8, 10)
    v.resize(20)
    for i in range(v.size):
        if i < 10:
            assert i in v
            assert v.get(i) == 0
        else:
            assert i not in v
            assert v.get(i) is None
