from pygraphblas import *


def test_type_lookup_name():
    assert types.Type.gb_from_name("INT8") == types.INT8.gb_type
