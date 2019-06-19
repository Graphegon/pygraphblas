
from _pygraphblas import lib, ffi
lib.GrB_init(lib.GrB_NONBLOCKING)

class GraphBLASException(Exception):
    pass

def _check(res):
    if res != lib.GrB_SUCCESS:
        raise GraphBLASException(res)
