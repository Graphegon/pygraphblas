import pytest
from pygraphblas import *


def test_type_lookup_name():
    assert types.Type.gb_from_name("INT8") == types.INT8.gb_type


def test_gb_from_type():
    with pytest.raises(TypeError):
        types._gb_from_type("")
