import sys
from itertools import repeat

from pygraphblas import Vector, Matrix, semiring
from pygraphblas.base import lib

def test_vector_create_from_type():
    m = Vector.from_type(int)
    assert m.size == 0
    assert m.nvals == 0
    m = Vector.from_type(int, 10)
    assert m.size == 10

def test_vector_gb_type():
    v = Vector.from_type(bool, 10)
    assert v.gb_type == lib.GrB_BOOL
    v = Vector.from_type(int, 10)
    assert v.gb_type == lib.GrB_INT64
    v = Vector.from_type(float, 10)
    assert v.gb_type == lib.GrB_FP64

def test_vector_set_element():
    m = Vector.from_type(int, 10)
    m[3] = 3
    assert m.size == 10
    assert m.nvals == 1
    assert m[3] == 3
    m = Vector.from_type(bool, 10)
    m[3] = True
    assert m.size == 10
    assert m.nvals == 1
    assert m[3] == True
    m = Vector.from_type(float, 10)
    m[3] = 3.3
    assert m.size == 10
    assert m.nvals == 1
    assert m[3] == 3.3

def test_vector_create_dup():
    m = Vector.from_type(int, 10)
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
    assert v == w
    assert v != x

def test_vector_ewise_add():
    v = Vector.from_list(list(range(10)))
    w = Vector.from_list(list(range(10)))
    x = v.ewise_add(w)
    assert x == Vector.from_lists(
        list(range(10)),
        list(range(0, 20, 2)))
    z = v + w
    assert x == z
    v += w
    assert v == z

def test_vector_ewise_mult():
    v = Vector.from_list(list(range(10)))
    w = Vector.from_list(list(range(10)))
    x = v.ewise_mult(w)
    assert x == Vector.from_lists(
        list(range(10)),
        list(map(lambda x: x*x, list(range(10)))))
    z = v * w
    assert x == z
    v *= w
    assert v == z

def test_vector_reduce_bool():
    v = Vector.from_type(bool, 10)
    assert not v.reduce_bool()
    v[3] = True
    assert v.reduce_bool()

def test_vector_reduce_int():
    v = Vector.from_type(int, 10)
    r = v.reduce_int()
    assert type(r) is int
    assert r == 0
    v[3] = 3
    v[4] = 4
    assert v.reduce_int() == 7

def test_vector_reduce_float():
    v = Vector.from_type(float, 10)
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

def test_vector_assign():
    v = Vector.from_type(int, 10)
    assert v.nvals == 0
    w = Vector.from_lists(
        list(range(10)),
        list(range(10)))
    v[:] = w
    assert v == w

    v[1:] = w[9:1:-1]
    assert v == Vector.from_lists(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 9, 8, 7, 6, 5, 4, 3, 2, 1])

    w[9:1:-1] = v[9:1:-1]
    assert w == v

    v[:] = 3
    assert v == Vector.from_lists(
        list(range(10)),
        list(repeat(3, 10)))

    v[1:] = 0
    assert v == Vector.from_lists(
        list(range(10)),
        [3] + list(repeat(0, 9)))

def test_vxm():
    m = Matrix.from_lists(
        [0,1,2],
        [1,2,0],
        [1,2,3])
    v = Vector.from_lists(
        [0,1,2],
        [2,3,4])
    o = v.vxm(m)
    assert o == Vector.from_lists(
        [0, 1, 2],
        [12, 2, 6])

    assert v @ m == o
    v @= m
    assert v == o
