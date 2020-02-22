from .base import lib, _check
from textwrap import dedent
from operator import methodcaller, itemgetter
from functools import partial
import numba
from numba import cfunc, jit, carray, cffi_support
from pygraphblas import  lib, ffi as core_ffi
from pygraphblas.base import lazy_property
from cffi import FFI

__all__ = [
    'Type',
    'BOOL',
    'INT8',
    'INT16',
    'INT32',
    'INT64',
    'UINT8',
    'UINT16',
    'UINT32',
    'UINT64',
    'FP32',
    'FP64',
    'binop',
    ]

class classproperty(object):
    def __init__(self, f):
        self.f = f
    def __get__(self, obj, owner):
        return self.f(owner)

def gb_type_to_type(gb_type):
    return MetaType._gb_type_map[gb_type]

class MetaType(type):

    _gb_type_map = {}

    def __new__(meta, type_name, bases, attrs):
        if attrs.get('base', False):
            cls = super().__new__(meta, type_name, bases, attrs)
            return cls
        if 'members' in attrs:
            m = attrs['members']
            cls_ffi = FFI()
            cls_ffi.cdef(build_udt_def(type_name, m))
            t = core_ffi.new('GrB_Type*')
            _check(lib.GrB_Type_new(t, cls_ffi.sizeof(type_name)))
            cffi_support.map_type(cls_ffi.typeof(type_name), use_record_dtype=True)
            attrs['ffi'] = cls_ffi
            attrs['gb_type'] = t[0]
            attrs['C'] = type_name
            attrs['member_def'] = list(map(methodcaller('split'), m))
            attrs['base_name'] = 'UDT'
        else:
            attrs['ffi'] = core_ffi
            gb_type_name = type_name
        
        cls = super().__new__(meta, type_name, bases, attrs)
        meta._gb_type_map[cls.gb_type] = cls
        cls.ptr = cls.C + '*'
        cls.zero = getattr(cls, 'zero', core_ffi.NULL)
        cls.one = getattr(cls, 'one', core_ffi.NULL)
        get = partial(getattr, lib)
        cls.base_name = base_name = getattr(cls, 'base_name', cls.__name__)
        cls.Monoid_new = get('GrB_Monoid_new_{}'.format(base_name))
        cls.Matrix_setElement = get('GrB_Matrix_setElement_{}'.format(base_name))
        cls.Matrix_extractElement = get('GrB_Matrix_extractElement_{}'.format(base_name))
        cls.Matrix_extractTuples = get('GrB_Matrix_extractTuples_{}'.format(base_name))
        cls.Matrix_assignScalar = get('GrB_Matrix_assign_{}'.format(base_name))
        cls.Vector_setElement = get('GrB_Vector_setElement_{}'.format(base_name))
        cls.Vector_extractElement = get('GrB_Vector_extractElement_{}'.format(base_name))
        cls.Vector_extractTuples = get('GrB_Vector_extractTuples_{}'.format(base_name))
        cls.Vector_assignScalar = get('GrB_Vector_assign_{}'.format(base_name))
        cls.Scalar_setElement = get('GxB_Scalar_setElement_{}'.format(base_name))
        cls.Scalar_extractElement = get('GxB_Scalar_extractElement_{}'.format(base_name))
        return cls

    def new_monoid(cls, op, identity):
        monoid = core_ffi.new('GrB_Monoid[1]')
        if cls.base_name == 'UDT':
            i = cls.ffi.new(cls.ptr)
            i[0] = identity
            identity = i
        _check(cls.Monoid_new(monoid, op.binaryop, identity))
        return monoid

    def new_semiring(cls, monoid, op):
        from .semiring import Semiring
        semiring = core_ffi.new('GrB_Semiring[1]')
        _check(lib.GrB_Semiring_new(semiring, monoid[0], op.get_binaryop(core_ffi.NULL)))
        return Semiring('PLUS', 'TIMES', cls.__name__, semiring[0])

class Type(metaclass=MetaType):
    one = 1
    zero = 0
    base = True
    typecode = None

    @classmethod
    def from_value(cls, value):
        if cls.base_name != 'UDT':
            return value
        data = cls.ffi.new('%s[1]' % cls.__name__)
        for (_, name), val in zip(cls.member_def, value):
            setattr(data[0], name, val)
        return data

    @classmethod
    def to_value(cls, cdata):
        if cls.base_name != 'UDT':
            return cdata
        return tuple(getattr(cdata, name) for (_, name) in cls.member_def)

class BOOL(Type):
    gb_type = lib.GrB_BOOL
    C = '_Bool'
    one = True
    zero = False
    typecode = 'B'
    numba_t = numba.boolean

    @classproperty
    def PLUS(cls):
        return cls.LOR

    @classproperty
    def TIMES(cls):
        return cls.LAND

    @classproperty
    def PLUS_MONOID(cls):
        return cls.LOR_MONOID

    @classproperty
    def TIMES_MONOID(cls):
        return cls.LAND_MONOID

    @classproperty
    def PLUS_TIMES(cls):
        return cls.LOR_LAND

class INT8(Type):
    gb_type = lib.GrB_INT8
    C = 'int8_t'
    typecode = 'b'
    numba_t = numba.int8

class UINT8(Type):
    gb_type = lib.GrB_UINT8
    C =  'uint8_t'
    typecode = 'B'
    numba_t = numba.uint8
    
class INT16(Type):
    gb_type = lib.GrB_INT16
    C = 'int16_t'
    typecode = 'i'
    numba_t = numba.int16

class UINT16(Type):
    gb_type = lib.GrB_UINT16
    C = 'uint16_t'
    typecode = 'I'
    numba_t = numba.uint16

class INT32(Type):
    gb_type = lib.GrB_INT32
    C =  'int32_t'
    typecode = 'l'
    numba_t = numba.int32

class UINT32(Type):
    gb_type = lib.GrB_UINT32
    C =  'uint32_t'
    typecode = 'L'
    numba_t = numba.uint32

class INT64(Type):
    gb_type = lib.GrB_INT64
    C = 'int64_t'
    typecode = 'q'
    numba_t = numba.int64

class UINT64(Type):
    gb_type = lib.GrB_UINT64
    C = 'uint64_t'
    typecode = 'Q'
    numba_t = numba.uint64

class FP32(Type):
    one = 1.0
    zero = 0.0
    gb_type = lib.GrB_FP32
    C = 'float'
    typecode = 'f'
    numba_t = numba.float32

class FP64(Type):
    one = 1.0
    zero = 0.0
    gb_type = lib.GrB_FP64
    C = 'double'
    typecode = 'd'
    numba_t = numba.float64

# class Complex(Type):
#     gb_type = lib.LAGraph_Complex
#     C = 'double _Complex'
#     typecode = None
#     udt = True
#     add_op = lib.Complex_plus
#     mult_op = lib.Complex_times
#     eq_op = lib.Complex_eq
#     monoid = lib.Complex_plus_monoid
#     semiring = lib.Complex_plus_times

#     @classmethod
#     def from_value(cls, value):
#         return ffi.new(cls.ptr, value)

#     @classmethod
#     def to_value(cls, data):
#         return data

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
            if cls.base_name == 'UDT':
                cls.ffi.cdef(build_binop_def(cls_name, func_name, boolean))
                sig = cffi_support.map_type(
                    cls.ffi.typeof(binop_name(cls_name, func_name)),
                    use_record_dtype=True)
            else:
                sig = numba.void(cls.numba_t, cls.numba_t, cls.numba_t)
                
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
