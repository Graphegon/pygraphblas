import pytest
from pygraphblas import types


def test_type_lookup_name():
    assert types.Type.gb_from_name("INT8") == types.INT8._gb_type


def test_gb_from_type():
    with pytest.raises(TypeError):
        types._gb_from_type("")

    
