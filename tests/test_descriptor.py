import pytest
from pygraphblas import descriptor, lib


def test_descriptor():
    assert descriptor.T0 == descriptor.Descriptor(lib.GrB_INP0, lib.GrB_TRAN, "T0")
    assert descriptor.T1 != descriptor.T0
    assert descriptor.T1 in descriptor.CT1
    assert descriptor.CT1 == (descriptor.C & descriptor.T1)
