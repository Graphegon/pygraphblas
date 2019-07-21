from _pygraphblas import lib, ffi

NULL = ffi.NULL

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

_default_type_ops = {
    lib.GrB_INT64: (lib.GrB_PLUS_INT64,
                    lib.GrB_TIMES_INT64),
    lib.GrB_FP64: (lib.GrB_PLUS_FP64,
                   lib.GrB_TIMES_FP64),
    lib.GrB_BOOL: (lib.GrB_PLUS_BOOL,
                   lib.GrB_TIMES_BOOL),
    }

def _default_add_op(obj):
    return _default_type_ops(obj.gb_type)[0]

def _default_mul_op(obj):
    return _default_type_ops(obj.gb_type)[1]

def _check(res):
    if res != lib.GrB_SUCCESS:
        raise _error_codes[res](ffi.string(lib.GrB_error()))

def _check_no_val_key_error(res):
    if res != lib.GrB_SUCCESS:
        if res == lib.GrB_NO_VALUE:
            raise KeyError
        raise _error_codes[res](ffi.string(lib.GrB_error()))
    
def _gb_from_type(typ):
    if typ is int:
        return lib.GrB_INT64
    if typ is float:
        return lib.GrB_FP64
    if typ is bool:
        return lib.GrB_BOOL
    return typ

def _build_range(rslice, stop_val):
    if rslice is None or \
       (rslice.start is None and
        rslice.stop is None and
        rslice.step is None):
        return lib.GrB_ALL, 0, None

    start = rslice.start
    stop = rslice.stop
    step = rslice.step
    if start is None:
        start = 0
    if stop is None:
        stop = stop_val
    if step is None:
        size = (stop - start) + 1
        I = ffi.new('GrB_Index[2]',
                    [start, stop])
        ni = lib.GxB_RANGE
    elif step < 0:
        step = abs(step)
        if start < stop:
            size = 0
        else:
            size = int((start-stop)/step) + 1
        I = ffi.new('GrB_Index[3]',
                    [start, stop, step])
        ni = lib.GxB_BACKWARDS
    else:
        if start > stop or step == 0:
            size = 0
        else:
            size = int((stop - start)/step) + 1
        I = ffi.new('GrB_Index[3]',
                    [start, stop, step])
        ni = lib.GxB_STRIDE
    return I, ni, size

def _get_descriptor(inp0_trans=False):
    desc = ffi.new('GrB_Descriptor*')
    if inp0_trans:
        # transpose input to get row
        _check(lib.GrB_Descriptor_new(desc))
        _check(lib.GrB_Descriptor_set(
            desc[0],
            lib.GrB_INP0,
            lib.GrB_TRAN))
    else:
        desc[0] = NULL
    return desc
