from .base import lib, _check
from textwrap import dedent
from operator import methodcaller
from functools import partial
from numba import cfunc, jit, carray, cffi_support
from pygraphblas import  lib, ffi
from cffi import FFI

class MetaType(type):

    def __new__(meta, name, bases, members):
        cls = super().__new__(meta, name, bases, members)
#    def __init__(self, gb_type, c_name, type_name, add, mult, aidentity, identity):
        type_name = members['type_name']
        c_name = members['c_name']
        add = members['add']
        mult = members['mult']
        cls.ffi = ffi
        cls.gb_type = members['gb_type']
        cls.type_name = type_name
        cls.C = c_name
        cls.ptr = c_name + '*'
        cls.aidentity = members['aidentity']
        cls.identity = members['identity']
        get = partial(getattr, lib)
        cls.Matrix_setElement = get('GrB_Matrix_setElement_{}'.format(type_name))
        cls.Matrix_extractElement = get('GrB_Matrix_extractElement_{}'.format(type_name))
        cls.Matrix_extractTuples = get('GrB_Matrix_extractTuples_{}'.format(type_name))
        cls.Matrix_assignScalar = get('GrB_Matrix_assign_{}'.format(type_name))
        cls.Vector_setElement = get('GrB_Vector_setElement_{}'.format(type_name))
        cls.Vector_extractElement = get('GrB_Vector_extractElement_{}'.format(type_name))
        cls.Vector_extractTuples = get('GrB_Vector_extractTuples_{}'.format(type_name))
        cls.Vector_assignScalar = get('GrB_Vector_assign_{}'.format(type_name))
        cls.Scalar_setElement = get('GxB_Scalar_setElement_{}'.format(type_name))
        cls.Scalar_extractElement = get('GxB_Scalar_extractElement_{}'.format(type_name))
        cls.add_op = get('GrB_PLUS_{}'.format(type_name))
        cls.mult_op = get('GrB_TIMES_{}'.format(type_name))
        cls.eq_op = get('GrB_EQ_{}'.format(type_name))
        cls.monoid = get('GxB_{}_{}_MONOID'.format(add, type_name))
        cls.semiring = get('GxB_{}_{}_{}'.format(add, mult, type_name))
        cls.invert = get('GrB_MINV_{}'.format(type_name))
        cls.neg = get('GrB_AINV_{}'.format(type_name))
        cls.abs_ = get('GxB_ABS_{}'.format(type_name))
        cls.not_ = get('GxB_LNOT_{}'.format(type_name))
        cls.first = get('GrB_FIRST_{}'.format(type_name))
        cls.gt = get('GrB_GT_{}'.format(type_name))
        cls.lt = get('GrB_LT_{}'.format(type_name))
        cls.ge = get('GrB_GE_{}'.format(type_name))
        cls.le = get('GrB_LE_{}'.format(type_name))
        cls.ne = get('GrB_NE_{}'.format(type_name))
        cls.eq = get('GrB_EQ_{}'.format(type_name))
        return cls
        

class Type(metaclass=MetaType):

    def from_value(self, value):
        return value

    def data_to_value(self, data):
        return data

    def ptr_to_value(self, data):
        return data[0]

class BOOL(Type):
    gb_type = lib.GrB_BOOL
    c_name = '_Bool'
    type_name = 'BOOL'
    add = 'LOR'
    mult = 'LAND'
    aidentity = True
    identity = False
    
# INT8 = Type(lib.GrB_INT8, 'int8_t', 'INT8', 'PLUS', 'TIMES', 1, 0)
# UINT8 = Type(lib.GrB_UINT8, 'uint8_t', 'UINT8', 'PLUS', 'TIMES', 1, 0)
# INT16 = Type(lib.GrB_INT16, 'int16_t', 'INT16', 'PLUS', 'TIMES', 1, 0)
# UINT16 = Type(lib.GrB_UINT16, 'uint16_t', 'UINT16', 'PLUS', 'TIMES', 1, 0)
# INT32 = Type(lib.GrB_INT32, 'int32_t', 'INT32', 'PLUS', 'TIMES', 1, 0)
# UINT32 = Type(lib.GrB_UINT32, 'uint32_t', 'UINT32', 'PLUS', 'TIMES', 1, 0)
# INT64 = Type(lib.GrB_INT64, 'int64_t', 'INT64', 'PLUS', 'TIMES', 1, 0)
# UINT64 = Type(lib.GrB_UINT64, 'uint64_t', 'UINT64', 'PLUS', 'TIMES', 1, 0)
# FP32 = Type(lib.GrB_FP32, 'float', 'FP32', 'PLUS', 'TIMES', 1.0, 0.0)
# FP64 = Type(lib.GrB_FP64, 'double', 'FP64', 'PLUS', 'TIMES', 1.0, 0.0)

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

def build_binop_def(typ, name, boolean=False):
    if boolean:
        return dedent("""
        typedef void (*{0})(bool*, {1}*, {1}*);
        """.format(binop_name(typ, name), typ))
    return dedent("""
    typedef void (*{0})({1}*, {1}*, {1}*);
    """.format(binop_name(typ, name), typ))

class UDT(Type):

    def __init__(self, type_name, members, aidentity=None, identity=None):
        self.ffi = FFI()
        self.type_name = type_name
        self.members = list(map(methodcaller('split'), members))
        self.ffi.cdef(build_udt_def(type_name, members))
        t = ffi.new('GrB_Type*')
        _check(lib.GrB_Type_new(t, self.ffi.sizeof(type_name)))
        cffi_support.map_type(self.ffi.typeof(type_name), use_record_dtype=True)
        self.gb_type = t[0]
        self.C = type_name
        self.ptr = type_name + '*'
        self.aidentity = aidentity
        self.identity = identity
        get = partial(getattr, lib)
        self.Matrix_setElement = lib.GrB_Matrix_setElement_UDT
        self.Matrix_extractElement = lib.GrB_Matrix_extractElement_UDT
        self.Matrix_extractTuples = lib.GrB_Matrix_extractTuples_UDT
        self.Matrix_assignScalar = lib.GrB_Matrix_assign_UDT
        self.Vector_setElement = lib.GrB_Vector_setElement_UDT
        self.Vector_extractElement = lib.GrB_Vector_extractElement_UDT
        self.Vector_extractTuples = lib.GrB_Vector_extractTuples_UDT
        self.Vector_assignScalar = lib.GrB_Vector_assign_UDT
        self.Scalar_setElement = lib.GxB_Scalar_setElement_UDT
        self.Scalar_extractElement = lib.GxB_Scalar_extractElement_UDT
        self.add_op = self.ffi.NULL
        self.mult_op = self.ffi.NULL
        self.eq_op = self.ffi.NULL
        self.monoid = self.ffi.NULL
        self.semiring = self.ffi.NULL

    def binop(self, binding, boolean=False):
        from .binaryop import BinaryOp
        def inner(func):
            self.ffi.cdef(build_binop_def(self.type_name, func.__name__, boolean))
            sig = cffi_support.map_type(
                self.ffi.typeof(binop_name(self.type_name, func.__name__)), 
                use_record_dtype=True)
            jitfunc = jit(func, nopython=True)
            @cfunc(sig)
            def wrapper(z_, x_, y_):            
                z = carray(z_, 1)[0]
                x = carray(x_, 1)[0]
                y = carray(y_, 1)[0]
                jitfunc(z, x, y)
            op = BinaryOp(func.__name__, self.type_name, wrapper, self, boolean)
            setattr(self, binding, op.get_binaryop())
            return op
        return inner

    def new_monoid(self, op, identity):
        monoid = ffi.new('GrB_Monoid[1]')
        _check(lib.GrB_Monoid_new_UDT(monoid, op.binaryop, identity))
        return monoid

    def new_semiring(self, monoid, op):
        from .semiring import Semiring
        semiring = ffi.new('GrB_Semiring[1]')
        _check(lib.GrB_Semiring_new(semiring, monoid[0], op.binaryop))
        return Semiring('add', 'mult', self.type_name, semiring[0])

    def from_value(self, value):
        data = self.ffi.new('%s[1]' % self.type_name)
        for (_, name), val in zip(self.members, value):
            setattr(data[0], name, val)
        return data
    
    def data_to_value(self, cdata):
        return tuple(getattr(cdata, name) for (_, name) in self.members)

    def ptr_to_value(self, cdata):
        cdata = cdata[0]
        return tuple(getattr(cdata, name) for (_, name) in self.members)

def udt(cls):
    name = cls.__name__
    members = cls.members
    aidentity = getattr(cls, 'aidentity', None)
    identity = getattr(cls, 'identity', None)
    new_udt = UDT(name, members, aidentity, identity)
    return new_udt
