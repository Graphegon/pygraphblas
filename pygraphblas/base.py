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


def _check(res, raise_no_val=False):
    if res != lib.GrB_SUCCESS:
        if raise_no_val and res == lib.GrB_NO_VALUE:
            raise KeyError
        raise _error_codes[res](ffi.string(lib.GrB_error()))


def _build_range(rslice, stop_val):
    # if already a list, return it and its length
    if isinstance(rslice, list):
        return rslice, len(rslice), len(rslice)

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
        "+": funcs.PLUS,
        "-": funcs.MINUS,
        "*": funcs.TIMES,
        "/": funcs.DIV,
    }[op]
