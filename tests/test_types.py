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
    assert order = []
    
