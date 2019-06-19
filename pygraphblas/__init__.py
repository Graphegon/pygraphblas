
from _pygraphblas import lib, ffi
lib.GrB_init(lib.GrB_NONBLOCKING)

class GraphBLASException(Exception):
    pass

def _check(res):
    if res != lib.GrB_SUCCESS:
        raise GraphBLASException(res)

def _gb_from_type(typ):
    if typ is int:
        return lib.GrB_INT64
    if typ is float:
        return lib.GrB_FP64
    if typ is bool:
        return lib.GrB_BOOL
    return typ

def _cffi_type_from(typ):
    if typ is int:
        return 'int*'
    if typ is float:
        return 'float*'
    if typ is bool:
        return 'bool*'
    raise TypeError('Unknown type to map to cffi')
