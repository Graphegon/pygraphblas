import sys
import re
import contextvars
from itertools import chain
from collections import defaultdict
import numba

from .base import lib, ffi, _gb_from_name, _check
from . import types

current_accum = contextvars.ContextVar('current_accum')
current_binop = contextvars.ContextVar('current_binop')

class BinaryOp:

    _auto_binaryops = defaultdict(dict)

    __slots__ = ('name', 'binaryop', 'token')

    def __init__(self, op, typ, binaryop, udt=None, boolean=False):
        if udt is not None:
            o = ffi.new('GrB_BinaryOp*')
            udt = udt.gb_type
            lib.GrB_BinaryOp_new(
                o,
                ffi.cast('GxB_binary_function', binaryop.address),
                lib.GrB_BOOL if boolean else udt, udt, udt)
            self.binaryop = o[0]
        else:
            self.binaryop = binaryop
            self.__class__._auto_binaryops[op][_gb_from_name(typ)] = binaryop
            cls = getattr(types, typ, None)
            if cls is not None:
                setattr(cls, op, self)
        self.name = '_'.join((op, typ))
        self.token = None

    def __enter__(self):
        self.token = current_binop.set(self)
        return self

    def __exit__(self, *errors):
        current_binop.reset(self.token)
        return False

    def get_binaryop(self, left=None, right=None):
        return self.binaryop

class AutoBinaryOp(BinaryOp):

    def __init__(self, name):
        self.name = name
        self.token = None

    def get_binaryop(self, left=None, right=None):
        typ = types.promote(left, right)
        return BinaryOp._auto_binaryops[self.name][typ.gb_type]

class Accum:

    __slots__ = ('binaryop', 'token')

    def __init__(self, binaryop):
        self.binaryop = binaryop

    def __enter__(self):
        self.token = current_accum.set(self.binaryop)
        return self

    def __exit__(self, *errors):
        current_accum.reset(self.token)
        return False

    def get_binaryop(self, operand1=None, operand2=None):
        return self.binaryop.get_binaryop(operand1)

__all__ = ['BinaryOp', 'AutoBinaryOp', 'Accum', 'current_binop', 'current_accum', 'binary_op']

grb_binop_re = re.compile(
    '^(GrB|GxB)_(FIRST|SECOND|MIN|MAX|PLUS|MINUS|RMINUS|TIMES|DIV|RDIV|'
    'PAIR|ANY|POW|EQ|NE|GT|LT|GE|LE|LOR|LAND|LXOR|BOR|BAND|BXOR|BXNOR|'
    'ATAN2|HYPOT|FMOD|REMAINDER|LDEXP|COPYSIGN|BGET|BSET|BCLR|BSHIFT|CMPLX)_'
    '(BOOL|UINT8|UINT16|UINT32|UINT64|INT8|INT16|INT32|INT64|FP32|FP64)$')

pure_bool_re = re.compile('^(GrB|GxB)_(LOR|LAND|LXOR)_(BOOL)$')

def binop_group(reg):
    srs = []
    for n in filter(None, [reg.match(i) for i in dir(lib)]):
        prefix, op, typ = n.groups()
        srs.append(BinaryOp(op, typ, getattr(lib, n.string)))
    return srs

def build_binaryops():
    this = sys.modules[__name__]
    for r in chain(binop_group(grb_binop_re), binop_group(pure_bool_re)):
        setattr(this, r.name, r)
    for name in BinaryOp._auto_binaryops:
        bo = AutoBinaryOp(name)
        setattr(this, name, bo)
        
def binary_op(arg_type, result_type=None):
    if result_type is None:
        result_type = arg_type
    def inner(func):
        func_name = func.__name__
        sig = numba.void(numba.types.CPointer(numba.boolean)
                         if result_type is types.BOOL else
                         numba.types.CPointer(arg_type.numba_t),
                         numba.types.CPointer(arg_type.numba_t),
                         numba.types.CPointer(arg_type.numba_t))
        jitfunc = numba.jit(func, nopython=True)
        @numba.cfunc(sig, nopython=True)
        def wrapper(z, x, y):
            result = jitfunc(x[0], y[0])
            z[0] = result

        out = ffi.new('GrB_BinaryOp*')
        lib.GrB_BinaryOp_new(
            out,
            ffi.cast('GxB_binary_function', wrapper.address),
            result_type.gb_type,
            arg_type.gb_type,
            arg_type.gb_type)

        return BinaryOp(func_name, arg_type.C, out[0])
    return inner

