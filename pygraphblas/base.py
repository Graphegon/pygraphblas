from _pygraphblas import lib, ffi
from numba import njit

__all__ = [
    "lib",
    "ffi",
    "NULL",
    "GraphBLASException",
    "NoValue",
    "UninitializedObject",
    "InvalidObject",
    "NullPointer",
    "InvalidObject",
    "NullPointer",
    "InvalidValue",
    "InvalidIndex",
    "DomainMismatch",
    "DimensionMismatch",
    "OutputNotEmpty",
    "OutOfMemory",
    "InsufficientSpace",
    "IndexOutOfBound",
    "Panic",
    "options_set",
]

NULL = ffi.NULL


def options_set(nthreads=None, chunk=None, burble=None):
    if nthreads is not None:
        nthreads = ffi.cast("int", nthreads)
        _check(lib.GxB_Global_Option_set(lib.GxB_GLOBAL_NTHREADS, nthreads))
    if chunk is not None:
        chunk = ffi.cast("double", chunk)
        _check(lib.GxB_Global_Option_set(lib.GxB_GLOBAL_CHUNK, chunk))
    if burble is not None:
        burble = ffi.cast("int", burble)
        _check(lib.GxB_Global_Option_set(lib.GxB_BURBLE, burble))


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
    1: NoValue,
    2: UninitializedObject,
    3: InvalidObject,
    4: NullPointer,
    5: InvalidValue,
    6: InvalidIndex,
    7: DomainMismatch,
    8: DimensionMismatch,
    9: OutputNotEmpty,
    10: OutOfMemory,
    11: InsufficientSpace,
    12: IndexOutOfBound,
    13: Panic,
}

_default_type_ops = {
    lib.GrB_INT64: (lib.GrB_PLUS_INT64, lib.GrB_TIMES_INT64),
    lib.GrB_FP64: (lib.GrB_PLUS_FP64, lib.GrB_TIMES_FP64),
    lib.GrB_BOOL: (lib.GrB_PLUS_BOOL, lib.GrB_TIMES_BOOL),
}


def _check(res):
    if res != lib.GrB_SUCCESS:
        raise _error_codes[res](ffi.string(lib.GrB_error()))


def _check_no_val_key_error(res):
    if res != lib.GrB_SUCCESS:
        if res == lib.GrB_NO_VALUE:
            raise KeyError
        raise _error_codes[res](ffi.string(lib.GrB_error()))


def _gb_from_name(name):
    name = name.lower()
    if name == "bool":
        return lib.GrB_BOOL
    if name == "uint8":
        return lib.GrB_UINT8
    if name == "int8":
        return lib.GrB_INT8
    if name == "uint16":
        return lib.GrB_UINT16
    if name == "int16":
        return lib.GrB_INT16
    if name == "uint32":
        return lib.GrB_UINT32
    if name == "int32":
        return lib.GrB_INT32
    if name == "uint64":
        return lib.GrB_UINT64
    if name == "int64":
        return lib.GrB_INT64
    if name == "fp32":
        return lib.GrB_FP32
    if name == "fp64":
        return lib.GrB_FP64
    if name == "fc32":
        return lib.GxB_FC32
    if name == "fc64":
        return lib.GxB_FC64
    raise TypeError("No such type %s" % name)


def _build_range(rslice, stop_val):
    # if already a list, return it and its length
    if isinstance(rslice, list):
        return rslice, len(rslice), len(rslice)

    if isinstance(rslice, int):
        return ffi.new("GrB_Index[1]", [rslice]), 1, 1

    if rslice is None or (
        rslice.start is None and rslice.stop is None and rslice.step is None
    ):
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
        I = ffi.new("GrB_Index[2]", [start, stop])
        ni = lib.GxB_RANGE
    elif step < 0:
        step = abs(step)
        if start < stop:
            size = 0
        else:
            size = int((start - stop) / step) + 1
        I = ffi.new("GrB_Index[3]", [start, stop, step])
        ni = lib.GxB_BACKWARDS
    else:
        if start > stop or step == 0:
            size = 0
        else:
            size = int((stop - start) / step) + 1
        I = ffi.new("GrB_Index[3]", [start, stop, step])
        ni = lib.GxB_STRIDE
    return I, ni, size


def _get_select_op(op):
    return {
        ">": lib.GxB_GT_THUNK,
        "<": lib.GxB_LT_THUNK,
        ">=": lib.GxB_GE_THUNK,
        "<=": lib.GxB_LE_THUNK,
        "!=": lib.GxB_NE_THUNK,
        "==": lib.GxB_EQ_THUNK,
        ">0": lib.GxB_GT_ZERO,
        "<0": lib.GxB_LT_ZERO,
        ">=0": lib.GxB_GE_ZERO,
        "<=0": lib.GxB_LE_ZERO,
        "!=0": lib.GxB_NONZERO,
        "==0": lib.GxB_EQ_ZERO,
    }[op]


def _get_bin_op(op, funcs):
    return {
        ">": funcs.GT,
        "<": funcs.LT,
        ">=": funcs.GE,
        "<=": funcs.LE,
        "!=": funcs.NE,
        "==": funcs.EQ,
    }[op]
