import pytest
from pygraphblas import *
from pygraphblas import types

def test_type_lookup_name():
    assert types.Type.gb_from_name("INT8") == types.INT8._gb_type


def test_gb_from_type():
    with pytest.raises(TypeError):
        types._gb_from_type("")

def test_promotion():
    order = []
    for t1 in (BOOL,) + types._int_types + types._float_types:
        for t2 in (BOOL,) + types._int_types + types._float_types:
            order.append(types.promote(t1, t2))
    assert [o.__name__ for o in order] == [
        'BOOL', 'INT8', 'INT16', 'INT32', 'INT64', 'UINT8', 'UINT16', 'UINT32', 'UINT64', 'FP32', 'FP64',
        'INT8', 'INT8', 'INT16', 'INT32', 'INT64', 'INT8', 'UINT16', 'UINT32', 'UINT64', 'FP32', 'FP64',
        'INT16', 'INT16', 'INT16', 'INT32', 'INT64', 'INT16', 'INT16', 'UINT32', 'UINT64', 'FP32', 'FP64',
        'INT32', 'INT32', 'INT32', 'INT32', 'INT64', 'INT32', 'INT32', 'INT32', 'UINT64', 'FP32', 'FP64',
        'INT64', 'INT64', 'INT64', 'INT64', 'INT64', 'INT64', 'INT64', 'INT64', 'INT64', 'FP32', 'FP64',
        'UINT8', 'INT8', 'INT16', 'INT32', 'INT64', 'UINT8', 'UINT16', 'UINT32', 'UINT64', 'FP32', 'FP64',
        'UINT16', 'UINT16', 'INT16', 'INT32', 'INT64', 'UINT16', 'UINT16', 'UINT32', 'UINT64', 'FP32', 'FP64',
        'UINT32', 'UINT32', 'UINT32', 'INT32', 'INT64', 'UINT32', 'UINT32', 'UINT32', 'UINT64', 'FP32', 'FP64',
        'UINT64', 'UINT64', 'UINT64', 'UINT64', 'INT64', 'UINT64', 'UINT64', 'UINT64', 'UINT64', 'FP32', 'FP64',
        'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP64',
        'FP64', 'FP64', 'FP64', 'FP64', 'FP64', 'FP64', 'FP64', 'FP64', 'FP64', 'FP64', 'FP64']
