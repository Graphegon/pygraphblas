from pygraphblas import *

def test_type_lookup_name():
    assert types.gb_from_name('int8') == types.INT8.gb_type
    
