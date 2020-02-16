import re, sys
from itertools import chain
from textwrap import dedent
from cffi import FFI
import numba
from numba import cfunc, jit, carray, cffi_support
import contextvars
from collections import defaultdict

from .base import lib, ffi as core_ffi, _gb_from_name, _check
from . import types

current_uop = contextvars.ContextVar('current_uop')

class UnaryOp:

    _auto_unaryops = defaultdict(dict)
    
    __slots__ = ('name', 'unaryop', 'ffi', 'token')

    def __init__(self, name, typ, op):
        self.name = '_'.join((name, typ))
        self.unaryop = op
        self.token = None
        self.__class__._auto_unaryops[name][_gb_from_name(typ)] = op
        cls = getattr(types, typ, None)
        if cls is not None:
            setattr(cls, name, self)

    def __enter__(self):
        self.token = current_uop.set(self)
        return self

    def __exit__(self, *errors):
        current_uop.reset(self.token)
        return False

    def get_unaryop(self, operand1=None):
        return self.unaryop

class AutoUnaryOp(UnaryOp):

    def __init__(self, name):
        self.name = name
        self.token = None

    def get_unaryop(self, operand1=None):
        return UnaryOp._auto_unaryops[self.name][operand1.gb_type]

__all__ = ['UnaryOp', 'AutoUnaryOp', 'unary_op', 'current_uop']

grb_uop_re = re.compile(
    '^GrB_(IDENTITY|AINV|MINV|LNOT|ONE|ABS)_'
    '(BOOL|UINT8|UINT16|UINT32|UINT64|INT8|INT16|INT32|INT64|FP32|FP64)$')

gxb_uop_re = re.compile(
    '^GxB_(ONE|ABS)_'
    '(BOOL|UINT8|UINT16|UINT32|UINT64|INT8|INT16|INT32|INT64|FP32|FP64)$')

bool_uop_re = re.compile('^GrB_LNOT$')

def uop_group(reg):
    srs = []
    for n in filter(None, [reg.match(i) for i in dir(lib)]):
        op, typ = n.groups()
        srs.append(UnaryOp(op, typ, getattr(lib, n.string)))
    return srs

def build_unaryops():
    this = sys.modules[__name__]
    for r in chain(uop_group(grb_uop_re), uop_group(gxb_uop_re)):
        setattr(this, r.name, r)
        __all__.append(r.name)
    for name in UnaryOp._auto_unaryops:
        bo = AutoUnaryOp(name)
        setattr(this, name, bo)
        __all__.append(name)
        

def _uop_name(name):
    return '_{0}_uop_function'.format(name)

def _build_uop_def(name, arg_type, result_type):
    decl = dedent("""
    typedef void (*{0})({1}*, {1}*);
    """.format(_uop_name(name), arg_type, result_type))
    return decl

def unary_op(arg_type, result_type=None, boolean=False):
    if result_type is None:
        result_type = arg_type
    def inner(func):
        func_name = func.__name__
        sig = numba.void(numba.types.CPointer(numba.boolean)
                         if boolean else numba.types.CPointer(arg_type.numba_t),
                         numba.types.CPointer(arg_type.numba_t))
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

        return UnaryOp(func_name, arg_type.c_name, out[0])
    return inner

