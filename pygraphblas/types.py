"""Pythonic GraphBLAS type wrappers including User Defined Types.

GraphBLAS has 13 built-in scalar types: Boolean, single and double
precision floating-point (real and complex), and 8, 16, 32, and 64-bit
signed and unsigned integers.  In addition, user-defined scalar types
can be created from nearly any C `typedef`, as long as the entire type
fits in a fixed-size contiguous block of memory (of arbitrary size).
All of these types can be used to create GraphBLAS sparse matrices,
vectors, or scalars.

"""
from .base import lib, _check, ffi as core_ffi
from textwrap import dedent
from operator import methodcaller, itemgetter
from functools import partial
import numba
import numpy
from numba import cfunc, jit, carray
from numba.core.typing import cffi_utils as cffi_support

from cffi import FFI

__pdoc__ = {}

__all__ = [
    "Type",
    "BOOL",
    "INT8",
    "INT16",
    "INT32",
    "INT64",
    "UINT8",
    "UINT16",
    "UINT32",
    "UINT64",
    "FP32",
    "FP64",
    "FC32",
    "FC64",
    "binop",
    "promote",
]


def _gb_type_to_type(gb_type):
    return MetaType._gb_type_map[gb_type]


class MetaType(type):

    _gb_type_map = {}
    _type_gb_map = {}
    _gb_name_type_map = {}
    _dtype_gb_map = {}

    def __new__(meta, type_name, bases, attrs):
        if attrs.get("base", False):
            cls = super().__new__(meta, type_name, bases, attrs)
            return cls
        if "members" in attrs:  # pragma: nocover
            m = attrs["members"]
            cls_ffi = FFI()
            cls_ffi.cdef(build_udt_def(type_name, m))
            t = core_ffi.new("GrB_Type*")
            _check(lib.GrB_Type_new(t, cls_ffi.sizeof(type_name)))
            cffi_support.map_type(cls_ffi.typeof(type_name), use_record_dtype=True)
            attrs["_ffi"] = cls_ffi
            attrs["_gb_type"] = t[0]
            attrs["_c_type"] = type_name
            attrs["member_def"] = list(map(methodcaller("split"), m))
            attrs["base_name"] = "UDT"
        else:
            attrs["_ffi"] = core_ffi
            _gb_type_name = type_name

        cls = super().__new__(meta, type_name, bases, attrs)
        meta._gb_type_map[cls._gb_type] = cls
        meta._type_gb_map[cls] = cls._gb_type
        meta._dtype_gb_map[cls._numpy_t] = cls
        meta._gb_name_type_map[type_name] = cls
        meta._gb_name_type_map[cls._c_type] = cls

        cls._ptr = cls._c_type + "*"
        cls.default_zero = getattr(cls, "default_zero", core_ffi.NULL)
        cls.default_one = getattr(cls, "default_one", core_ffi.NULL)
        get = partial(getattr, lib)
        cls._base_name = base_name = getattr(cls, "_base_name", cls.__name__)
        cls._prefix = prefix = getattr(cls, "_prefix", "GrB")
        cls._Monoid_new = get("{}_Monoid_new_{}".format(prefix, base_name))
        cls._Matrix_setElement = get(
            "{}_Matrix_setElement_{}".format(prefix, base_name)
        )
        cls._Matrix_extractElement = get(
            "{}_Matrix_extractElement_{}".format(prefix, base_name)
        )
        cls._Matrix_extractTuples = get(
            "{}_Matrix_extractTuples_{}".format(prefix, base_name)
        )
        cls._Matrix_assignScalar = get("{}_Matrix_assign_{}".format(prefix, base_name))
        cls._Matrix_apply_BinaryOp1st = get(
            "{}_Matrix_apply_BinaryOp1st_{}".format(prefix, base_name)
        )
        cls._Matrix_apply_BinaryOp2nd = get(
            "{}_Matrix_apply_BinaryOp2nd_{}".format(prefix, base_name)
        )
        cls._Vector_setElement = get(
            "{}_Vector_setElement_{}".format(prefix, base_name)
        )
        cls._Vector_extractElement = get(
            "{}_Vector_extractElement_{}".format(prefix, base_name)
        )
        cls._Vector_extractTuples = get(
            "{}_Vector_extractTuples_{}".format(prefix, base_name)
        )
        cls._Vector_assignScalar = get("{}_Vector_assign_{}".format(prefix, base_name))
        cls._Vector_apply_BinaryOp1st = get(
            "{}_Vector_apply_BinaryOp1st_{}".format(prefix, base_name)
        )
        cls._Vector_apply_BinaryOp2nd = get(
            "{}_Vector_apply_BinaryOp2nd_{}".format(prefix, base_name)
        )
        cls._Scalar_setElement = get("GxB_Scalar_setElement_{}".format(base_name))
        cls._Scalar_extractElement = get(
            "GxB_Scalar_extractElement_{}".format(base_name)
        )
        return cls

    def new_monoid(cls, op, identity):
        from .monoid import Monoid

        monoid = core_ffi.new("GrB_Monoid[1]")
        if cls._base_name == "UDT":  # pragma: nocover
            i = cls._ffi.new(cls._ptr)
            i[0] = identity
            identity = i
        _check(cls._Monoid_new(monoid, op.binaryop, identity))
        return Monoid("PLUS", cls.__name__, monoid[0], udt=cls)

    def new_semiring(cls, monoid, op):
        from .semiring import Semiring

        semiring = core_ffi.new("GrB_Semiring[1]")
        _check(
            lib.GrB_Semiring_new(
                semiring, monoid.get_monoid(), op.get_binaryop(core_ffi.NULL)
            )
        )
        return Semiring("PLUS", "TIMES", cls.__name__, semiring[0], udt=cls)

    def gb_from_name(cls, name):
        return cls._gb_name_type_map[name]._gb_type

    @property
    def GrB_name(cls):
        return "GrB_" + cls.__name__

    @property
    def size(cls):
        s = cls._ffi.new("size_t*")
        _check(lib.GxB_Type_size(s, cls._gb_type))
        return s[0]


class Type(metaclass=MetaType):
    default_one = 1
    """The default value used to represent 1 for filling in types."""
    default_zero = 0
    """The default value used to represent 0 for filling in types."""
    base = True
    _typecode = None

    @classmethod
    def format_value(cls, val, width=2, prec=None):
        """Return the value as a formatted string for display."""
        return f"{val:{width}}"

    @classmethod
    def _default_addop(cls):
        return cls.PLUS

    @classmethod
    def _default_multop(cls):
        return cls.TIMES

    @classmethod
    def _default_semiring(cls):
        return cls.PLUS_TIMES

    @classmethod
    def _from_value(cls, value):
        """"""
        if cls._base_name != "UDT":
            return value
        else:  # pragma: nocover
            data = cls._ffi.new("%s[1]" % cls.__name__)
            for (_, name), val in zip(cls.member_def, value):
                setattr(data[0], name, val)
            return data

    @classmethod
    def _to_value(cls, cdata):
        if cls._base_name != "UDT":
            return cdata
        else:  # pragma: nocover
            return tuple(getattr(cdata, name) for (_, name) in cls.member_def)


class BOOL(Type):
    """GraphBLAS Boolean Type."""

    _gb_type = lib.GrB_BOOL
    _c_type = "_Bool"
    default_one = True
    default_zero = False
    _typecode = "B"
    _numba_t = numba.boolean
    _numpy_t = numpy.bool_

    @classmethod
    def _default_addop(self):  # pragma: nocover
        return self.LOR

    @classmethod
    def _default_multop(self):  # pragma: nocover
        return self.LAND

    @classmethod
    def _default_semiring(self):  # pragma: nocover
        return self.LOR_LAND

    @classmethod
    def format_value(cls, val, width=2, prec=None):
        f = "{:>%s}" % width
        if not isinstance(val, bool):
            return f.format(val)
        return f.format("t") if val is True else f.format("f")

    @classmethod
    def _to_value(cls, cdata):
        return bool(cdata)


class INT8(Type):
    """GraphBLAS 8 bit signed integer."""

    _gb_type = lib.GrB_INT8
    _c_type = "int8_t"
    _typecode = "b"
    _numba_t = numba.int8
    _numpy_t = numpy.int8


class UINT8(Type):
    """GraphBLAS 8 bit unsigned integer."""

    _gb_type = lib.GrB_UINT8
    _c_type = "uint8_t"
    _typecode = "B"
    _numba_t = numba.uint8
    _numpy_t = numpy.uint8


class INT16(Type):
    """GraphBLAS 16 bit signed integer."""

    _gb_type = lib.GrB_INT16
    _c_type = "int16_t"
    _typecode = "i"
    _numba_t = numba.int16
    _numpy_t = numpy.int16


class UINT16(Type):
    """GraphBLAS 16 bit unsigned integer."""

    _gb_type = lib.GrB_UINT16
    _c_type = "uint16_t"
    _typecode = "I"
    _numba_t = numba.uint16
    _numpy_t = numpy.uint16


class INT32(Type):
    """GraphBLAS 32 bit signed integer."""

    _gb_type = lib.GrB_INT32
    _c_type = "int32_t"
    _typecode = "l"
    _numba_t = numba.int32
    _numpy_t = numpy.int32


class UINT32(Type):
    """GraphBLAS 32 bit unsigned integer."""

    _gb_type = lib.GrB_UINT32
    _c_type = "uint32_t"
    _typecode = "L"
    _numba_t = numba.uint32
    _numpy_t = numpy.uint32


class INT64(Type):
    """GraphBLAS 64 bit signed integer."""

    _gb_type = lib.GrB_INT64
    _c_type = "int64_t"
    _typecode = "q"
    _numba_t = numba.int64
    _numpy_t = numpy.int64


class UINT64(Type):
    """GraphBLAS 64 bit unsigned integer."""

    _gb_type = lib.GrB_UINT64
    _c_type = "uint64_t"
    _typecode = "Q"
    _numba_t = numba.uint64
    _numpy_t = numpy.uint64


class FP32(Type):
    """GraphBLAS 32 bit float."""

    default_one = 1.0
    default_zero = 0.0
    _gb_type = lib.GrB_FP32
    _c_type = "float"
    _typecode = "f"
    _numba_t = numba.float32
    _numpy_t = numpy.float32

    @classmethod
    def format_value(cls, val, width=2, prec=2):
        return f"{val:>{width}.{prec}}"


class FP64(Type):
    """GraphBLAS 64 bit float."""

    default_one = 1.0
    default_zero = 0.0
    _gb_type = lib.GrB_FP64
    _c_type = "double"
    _typecode = "d"
    _numba_t = numba.float64
    _numpy_t = numpy.float64

    @classmethod
    def format_value(cls, val, width=2, prec=2):
        return f"{val:>{width}.{prec}}"


class FC32(Type):
    """GraphBLAS 32 bit float complex."""

    _prefix = "GxB"
    default_one = complex(1.0)
    default_zero = complex(0.0)
    _gb_type = lib.GxB_FC32
    _c_type = "float _Complex"
    _numba_t = numba.complex64
    _numpy_t = numpy.complex64


class FC64(Type):
    """GraphBLAS 64 bit float complex."""

    _prefix = "GxB"
    default_one = complex(1.0)
    default_zero = complex(0.0)
    _gb_type = lib.GxB_FC64
    _c_type = "double _Complex"
    _numba_t = numba.complex128
    _numpy_t = numpy.complex128


def _gb_from_type(typ):  # pragma: nocover
    if typ is int:
        return INT64
    if typ is float:
        return FP64
    if typ is bool:
        return BOOL
    if typ is complex:  # pragma: no cover
        return FC64
    raise TypeError(f'cannot turn {typ!r} into GraphBLAS type.')


def udt_head(name):  # pragma: nocover
    return dedent(
        """
    typedef struct %s {
    """
        % name
    )


def udt_body(members):  # pragma: nocover
    return ";\n".join(members) + ";"


def udt_tail(name):  # pragma: nocover
    return dedent(
        """
    } %s;
    """
        % name
    )


def build_udt_def(typ, members):  # pragma: nocover
    return udt_head(typ) + udt_body(members) + udt_tail(typ)


def binop_name(typ, name):  # pragma: nocover
    return "{0}_{1}_binop_function".format(typ, name)


def build_binop_def(typ, name, boolean=False):  # pragma: nocover
    if boolean:
        return dedent(
            """
        typedef void (*{0})(bool*, {1}*, {1}*);
        """.format(
                binop_name(typ, name), typ
            )
        )
    return dedent(
        """
    typedef void (*{0})({1}*, {1}*, {1}*);
    """.format(
            binop_name(typ, name), typ
        )
    )


def binop(boolean=False):  # pragma: nocover
    from .binaryop import BinaryOp

    class inner:
        def __init__(self, func):
            self.func = func

        def __set_name__(self, cls, name):
            func_name = self.func.__name__
            cls_name = cls.__name__
            if cls._base_name == "UDT":
                cls._ffi.cdef(build_binop_def(cls_name, func_name, boolean))
                sig = cffi_support.map_type(
                    cls._ffi.typeof(binop_name(cls_name, func_name)),
                    use_record_dtype=True,
                )
            else:
                sig = numba.void(cls._numba_t, cls._numba_t, cls._numba_t)

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


def get_add(sring):
    monoid = core_ffi.new("GrB_Monoid*")
    _check(lib.GxB_Semiring_add(monoid, sring))
    return monoid[0]


def get_binaryop(moid):
    op = core_ffi.new("GrB_BinaryOp*")
    _check(lib.GxB_Monoid_operator(op, moid))
    return op[0]


def get_ztype(bop):
    typ = core_ffi.new("GrB_Type*")
    _check(lib.GxB_BinaryOp_ztype(typ, bop))
    return _gb_type_to_type(typ[0])


def get_semiring_ztype(sring):
    return get_ztype(get_binaryop(get_add(sring)))


_int_types = (INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64)

_float_types = (FP32, FP64)

_promotion_order = (
    FC64,
    FC32,
    FP64,
    FP32,
    INT64,
    UINT64,
    INT32,
    UINT32,
    INT16,
    UINT16,
    INT8,
    UINT8,
)


def promote(left, right):
    """Do type promotion, determine the result type of an operation
    infered from the operands and possible semiring.

    """
    if left == right:
        return left
    elif left == BOOL:
        return right
    elif right == BOOL:
        return left
    for t in _promotion_order:
        if left == t or right == t:
            return t
    raise TypeError(
        "inconvertable types %s and %s" % (repr(left), repr(right))
    )  # pragma: nocover
