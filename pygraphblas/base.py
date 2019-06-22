from _pygraphblas import lib, ffi
lib.GrB_init(lib.GrB_NONBLOCKING)

class GraphBLASException(Exception):
    pass

class NoValue(GraphBLASException):
    pass

class UninitializedObject(GraphBLASException):
    pass

class InvalidObject(GraphBLASException):
    pass

class NullPointer(GraphBLASException):
    pass

class InvalidValue(GraphBLASException):
    pass

class InvalidIndex(GraphBLASException):
    pass

class DomainMismatch(GraphBLASException):
    pass

class DimensionMismatch(GraphBLASException):
    pass

class OutputNotEmpty(GraphBLASException):
    pass

class OutOfMemory(GraphBLASException):
    pass

class InsufficientSpace(GraphBLASException):
    pass

class IndexOutOfBound(GraphBLASException):
    pass

class Panic(GraphBLASException):
    pass

_error_codes = {
    1:  NoValue,
    2:  UninitializedObject,
    3:  InvalidObject,
    4:  NullPointer,
    5:  InvalidValue,
    6:  InvalidIndex,
    7:  DomainMismatch,
    8:  DimensionMismatch,
    9:  OutputNotEmpty,
    10: OutOfMemory,
    11: InsufficientSpace,
    12: IndexOutOfBound,
    13: Panic,
    }

def _check(res):
    if res != lib.GrB_SUCCESS:
        raise _error_codes[res]

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
