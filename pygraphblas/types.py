from .base import lib
from textwrap import dedent
from operator import methodcaller
from functools import partial
from numba import cfunc, jit, carray, cffi_support
from pygraphblas import  lib, ffi
from cffi import FFI

class Type:
    def __init__(self, gb_type, c_name, type_suffix, add, mult, aidentity, identity):
        self.gb_type = gb_type
        self.C = c_name
        self.type_name = type_suffix
        self.aidentity = aidentity
        self.identity = identity
        get = partial(getattr, lib)
        self.Matrix_setElement = get('GrB_Matrix_setElement_{}'.format(type_suffix))
        self.Matrix_extractElement = get('GrB_Matrix_extractElement_{}'.format(type_suffix))
        self.Matrix_extractTuples = get('GrB_Matrix_extractTuples_{}'.format(type_suffix))
        self.Matrix_assignScalar = get('GrB_Matrix_assign_{}'.format(type_suffix))
        self.Vector_setElement = get('GrB_Vector_setElement_{}'.format(type_suffix))
        self.Vector_extractElement = get('GrB_Vector_extractElement_{}'.format(type_suffix))
        self.Vector_extractTuples = get('GrB_Vector_extractTuples_{}'.format(type_suffix))
        self.Vector_assignScalar = get('GrB_Vector_assign_{}'.format(type_suffix))
        self.Scalar_setElement = get('GxB_Scalar_setElement_{}'.format(type_suffix))
        self.Scalar_extractElement = get('GxB_Scalar_extractElement_{}'.format(type_suffix))
        self.add_op = get('GrB_PLUS_{}'.format(type_suffix))
        self.mult_op = get('GrB_TIMES_{}'.format(type_suffix))
        self.monoid = get('GxB_{}_{}_MONOID'.format(add, type_suffix))
        self.semiring = get('GxB_{}_{}_{}'.format(add, mult, type_suffix))
        self.invert = get('GrB_MINV_{}'.format(type_suffix))
        self.neg = get('GrB_AINV_{}'.format(type_suffix))
        self.abs_ = get('GxB_ABS_{}'.format(type_suffix))
        self.not_ = get('GxB_LNOT_{}'.format(type_suffix))
        self.first = get('GrB_FIRST_{}'.format(type_suffix))
        self.gt = get('GrB_GT_{}'.format(type_suffix))
        self.lt = get('GrB_LT_{}'.format(type_suffix))
        self.ge = get('GrB_GE_{}'.format(type_suffix))
        self.le = get('GrB_LE_{}'.format(type_suffix))
        self.ne = get('GrB_NE_{}'.format(type_suffix))
        self.eq = get('GrB_EQ_{}'.format(type_suffix))

    def from_value(self, value):
        return value

    def to_value(self, data):
        return data

BOOL = Type(lib.GrB_BOOL, '_Bool', 'BOOL', 'LOR', 'LAND', True, False)
INT8 = Type(lib.GrB_INT8, 'int8_t', 'INT8', 'PLUS', 'TIMES', 1, 0)
UINT8 = Type(lib.GrB_UINT8, 'uint8_t', 'UINT8', 'PLUS', 'TIMES', 1, 0)
INT16 = Type(lib.GrB_INT16, 'int16_t', 'INT16', 'PLUS', 'TIMES', 1, 0)
UINT16 = Type(lib.GrB_UINT16, 'uint16_t', 'INT16', 'PLUS', 'TIMES', 1, 0)
INT32 = Type(lib.GrB_INT32, 'int32_t', 'INT32', 'PLUS', 'TIMES', 1, 0)
UINT32 = Type(lib.GrB_UINT32, 'uint32_t', 'INT32', 'PLUS', 'TIMES', 1, 0)
INT64 = Type(lib.GrB_INT64, 'int64_t', 'INT64', 'PLUS', 'TIMES', 1, 0)
UINT64 = Type(lib.GrB_UINT64, 'uint64_t', 'INT64', 'PLUS', 'TIMES', 1, 0)
FP32 = Type(lib.GrB_FP32, 'float', 'FP32', 'PLUS', 'TIMES', 1.0, 0.0)
FP64 = Type(lib.GrB_FP64, 'double', 'FP64', 'PLUS', 'TIMES', 1.0, 0.0)

def _gb_from_type(typ):
    if typ is int:
        return INT64
    if typ is float:
        return FP64
    if typ is bool:
        return BOOL
    return typ

def udt_head(name):
    return dedent("""
    typedef struct %s {
    """ % name)

def udt_body(members):
    return ";\n".join(members) + ';'

def udt_tail(name):
    return dedent("""
    } %s;
    """ % name)

def build_udt_def(typ, members):
    return (udt_head(typ) +
            udt_body(members) +
            udt_tail(typ))

def binop_name(typ, name):
    return '{0}_{1}_binop_function'.format(typ, name)

def build_binop_def(typ, name):
    return dedent("""
    typedef void (*{0})({1}*, {1}*, {1}*);
    """.format(binop_name(typ, name), typ))

class UDT(Type):
    def __init__(self, type_name, members, add_func_name, mul_func_name=None):
        self.ffi = FFI()
        self.type_name = type_name
        self.members = map(methodcaller('split'), members)
        self.ffi.cdef(build_udt_def(type_name, members))
        t = ffi.new('GrB_Type*')
        lib.GrB_Type_new(t, self.ffi.sizeof(type_name))
        cffi_support.map_type(self.ffi.typeof(type_name), use_record_dtype=True)
        self.gb_type = t

    def binop(self, func_name):
        self.ffi.cdef(build_binop_def(self.type_name, func_name))
        sig = cffi_support.map_type(
            self.ffi.typeof(binop_name(self.type_name, func_name)), 
            use_record_dtype=True)    

        def inner(func):
            jitfunc = jit(func, nopython=True)
            @cfunc(sig)
            def wrapper(z_, x_, y_):            
                z = carray(z_, 1)[0]
                x = carray(x_, 1)[0]
                y = carray(y_, 1)[0]
                jitfunc(z, x, y)
            return wrapper
        return inner

    def semiring(self, add_func_name, mul_func_name):
        pass

    def from_value(self, *args):
        data = self.ffi.new('%s[1]' % self.type_name)
        for (_, name), val in zip(self.members, args):
            setattr(data[0], name, val)
        return data
    
    def to_value(self, ptr):
        return self.ffi.cast('%s*' % self.type_name, ptr)
