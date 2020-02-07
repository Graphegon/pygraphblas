import re, sys
from itertools import chain
from textwrap import dedent
from cffi import FFI
from numba import cfunc, jit, carray, cffi_support

from .base import lib, ffi as core_ffi

class UnaryOp:

    __slots__ = ('name', 'unaryop', 'ffi')

    def __init__(self, name, unaryop, ffi=None):
        self.name = name
        self.unaryop = unaryop
        self.ffi = ffi

    def __enter__(self):
        return self

    def __exit__(self, *errors):
        return False

__all__ = ['UnaryOp', 'uop']

grb_uop_re = re.compile(
    '^GrB_(IDENTITY|AINV|MINV|LNOT|ONE|ABS)_'
    '(BOOL|UINT8|UINT16|UINT32|UINT64|INT8|INT16|INT32|INT64|FP32|FP64)$')

bool_uop_re = re.compile('^GrB_LNOT$')

def uop_group(reg):
    return [UnaryOp(n.string[4:].lower(), getattr(lib, n.string))
            for n in filter(None, [reg.match(i) for i in dir(lib)])]

def build_unaryops():
    this = sys.modules[__name__]
    for r in chain(uop_group(grb_uop_re), uop_group(bool_uop_re)):
        setattr(this, r.name, r)
        __all__.append(r.name)

def _uop_name(name):
    return '_{0}_uop_function'.format(name)

def _build_uop_def(name, arg_type, result_type):
    decl = dedent("""
    typedef void (*{0})({1}*, {1}*);
    """.format(_uop_name(name), arg_type, result_type))
    return decl

def uop(arg_type, result_type=None):
    if result_type is None:
        result_type = arg_type
    def inner(func):
        func_name = func.__name__
        ffi = FFI()
        ffi.cdef(_build_uop_def(func_name, arg_type.c_name, result_type.c_name))
        sig = cffi_support.map_type(
            ffi.typeof(_uop_name(func_name)),
            use_record_dtype=True)
        jitfunc = jit(func, nopython=True)
        @cfunc(sig, nopython=True)
        def wrapper(z, x):
            result = jitfunc(x[0])
            z[0] = result

        out = core_ffi.new('GrB_UnaryOp*')
        lib.GrB_UnaryOp_new(
            out,
            core_ffi.cast('GxB_unary_function', wrapper.address),
            result_type.gb_type,
            arg_type.gb_type)
        
        return UnaryOp(func_name, out[0], ffi)
    return inner

