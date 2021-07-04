import pytest
from pygraphblas import *
from pygraphblas import lib


def test_descriptor():
    assert descriptor.T0 == descriptor.Descriptor(lib.GrB_DESC_T0, "T0")
    assert descriptor.T1 != descriptor.T0
    assert descriptor.T1 in descriptor.CT1
    assert descriptor.CT1 == (descriptor.C & descriptor.T1)


def test_RCT0():
    M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [True, True, True])
    w = Vector.sparse(BOOL, 3)
    v = Vector.sparse(BOOL, 3)

    w[0] = True
    M.mxv(w, out=w, mask=v, desc=descriptor.RCT0)
    assert w.iseq(Vector.from_lists([1], [True], 3))


def test_RC():
    M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [True, True, True])
    w = Vector.sparse(BOOL, 3)
    v = Vector.sparse(BOOL, 3)

    w[0] = True
    M.mxv(w, out=w, mask=v, desc=descriptor.RC)
    assert w.iseq(Vector.from_lists([2], [True], 3))
