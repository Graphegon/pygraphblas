from .base import lib, _check
from textwrap import dedent
from operator import methodcaller, itemgetter
from functools import partial
from numba import cfunc, jit, carray, cffi_support
from pygraphblas import  lib, ffi
from pygraphblas.base import lazy_property
from cffi import FFI

def gb_type_to_type(gb_type):
    return MetaType._gb_type_map[gb_type]

class MetaType(type):

    _gb_type_map = {}

    def __new__(meta, type_name, bases, members):
        if members.get('base', False):
            cls = super().__new__(meta, type_name, bases, members)
            return cls
        cls = super().__new__(meta, type_name, bases, members)
        meta._gb_type_map[cls.gb_type] = cls
        type_name = cls.__name__
        c_name = cls.c_name
        add = cls.add
        mult = cls.mult
        cls.ffi = ffi
        cls.C = c_name
        cls.ptr = c_name + '*'
        cls.aidentity = cls.aidentity
        cls.identity = cls.identity
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
    add = 'PLUS'
    mult = 'TIMES'
    aidentity = 1
    identity = 0
    base = True
    typecode = None

    @classmethod
    def from_value(cls, value):
        return value

    @classmethod
    def to_value(cls, data):
        return data

class BOOL(Type):
    gb_type = lib.GrB_BOOL
    c_name = '_Bool'
    add = 'LOR'
    mult = 'LAND'
    aidentity = True
    identity = False
    typecode = 'B'

class INT8(Type):
    gb_type = lib.GrB_INT8
    c_name = 'int8_t'
    typecode = 'b'

class UINT8(Type):
    gb_type = lib.GrB_UINT8
    c_name =  'uint8_t'
    typecode = 'B'

class INT16(Type):
    gb_type = lib.GrB_INT16
    c_name = 'int16_t'
    typecode = 'i'

class UINT16(Type):
    gb_type = lib.GrB_UINT16
    c_name = 'uint16_t'
    typecode = 'I'

class INT32(Type):
    gb_type = lib.GrB_INT32
    c_name =  'int32_t'
    typecode = 'l'

class UINT32(Type):
    gb_type = lib.GrB_UINT32
    c_name =  'uint32_t'
    typecode = 'L'

class INT64(Type):
    gb_type = lib.GrB_INT64
    c_name = 'int64_t'
    typecode = 'q'

class UINT64(Type):
    gb_type = lib.GrB_UINT64
    c_name = 'uint64_t'
    typecode = 'Q'

class FP32(Type):
    gb_type = lib.GrB_FP32
    c_name = 'float'
    typecode = 'f'

class FP64(Type):
    gb_type = lib.GrB_FP64
    c_name = 'double'
    typecode = 'd'

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

def binop(boolean=False):
    from .binaryop import BinaryOp
    class inner:

        def __init__(self, func):
            self.func = func

        def __set_name__(self, cls, name):
            func_name = self.func.__name__
            cls_name = cls.__name__
            cls.ffi.cdef(build_binop_def(cls_name, func_name, boolean))
            sig = cffi_support.map_type(
                cls.ffi.typeof(binop_name(cls_name, func_name)),
                use_record_dtype=True)
            jitfunc = jit(self.func, nopython=True)
            @cfunc(sig)
            def wrapper(z_, x_, y_):
                z = carray(z_, 1)[0]
                x = carray(x_, 1)[0]
                y = carray(y_, 1)[0]
                jitfunc(z, x, y)
            self.op = BinaryOp(func_name, cls_name, wrapper, cls, boolean)
            setattr(cls, func_name, self.op)

    return inner

class MetaStruct(type):

    def __new__(meta, type_name, bases, members):
        if members.get('base'):
            cls = super().__new__(meta, type_name, bases, members)
            return cls
        m = members['members']
        cls_ffi = members['ffi'] = FFI()
        cls_ffi.cdef(build_udt_def(type_name, m))
        t = ffi.new('GrB_Type*')
        _check(lib.GrB_Type_new(t, cls_ffi.sizeof(type_name)))
        cffi_support.map_type(cls_ffi.typeof(type_name), use_record_dtype=True)
        members['gb_type'] = t[0]
        
        cls = super().__new__(meta, type_name, bases, members)
        cls.member_def = list(map(methodcaller('split'), m))
        cls.C = type_name
        cls.ptr = type_name + '*'
        get = partial(getattr, lib)
        cls.Matrix_setElement = lib.GrB_Matrix_setElement_UDT
        cls.Matrix_extractElement = lib.GrB_Matrix_extractElement_UDT
        cls.Matrix_extractTuples = lib.GrB_Matrix_extractTuples_UDT
        cls.Matrix_assignScalar = lib.GrB_Matrix_assign_UDT
        cls.Vector_setElement = lib.GrB_Vector_setElement_UDT
        cls.Vector_extractElement = lib.GrB_Vector_extractElement_UDT
        cls.Vector_extractTuples = lib.GrB_Vector_extractTuples_UDT
        cls.Vector_assignScalar = lib.GrB_Vector_assign_UDT
        cls.Scalar_setElement = lib.GxB_Scalar_setElement_UDT
        cls.Scalar_extractElement = lib.GxB_Scalar_extractElement_UDT
        cls.identity = cls.from_value(cls.identity)
        for op_name in ['eq_op', 'add_op', 'mult_op']:
            if not hasattr(cls, op_name):
                setattr(cls, op_name, cls.ffi.NULL)
        return cls

    def new_monoid(cls, op, identity):
        monoid = ffi.new('GrB_Monoid[1]')
        _check(lib.GrB_Monoid_new_UDT(monoid, op.binaryop, identity))
        return monoid

    def new_semiring(cls, monoid, op):
        from .semiring import Semiring
        semiring = ffi.new('GrB_Semiring[1]')
        _check(lib.GrB_Semiring_new(semiring, monoid[0], op.binaryop))
        return Semiring('add', 'mult', cls.__name__, semiring[0])

    @lazy_property
    def monoid(cls):
        return cls.new_monoid(cls.add_op, cls.identity)

    @lazy_property
    def semiring(cls):
        return cls.new_semiring(cls.monoid, cls.mult_op)

class Struct(metaclass=MetaStruct):

    members = ()
    base = True

    @classmethod
    def from_value(cls, value):
        data = cls.ffi.new('%s[1]' % cls.__name__)
        for (_, name), val in zip(cls.member_def, value):
            setattr(data[0], name, val)
        return data

    @classmethod
    def to_value(cls, cdata):
        return tuple(getattr(cdata, name) for (_, name) in cls.member_def)
