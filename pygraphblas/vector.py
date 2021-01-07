import operator
import weakref
from array import array

from .base import (
    lib,
    ffi,
    NULL,
    NoValue,
    _check,
    _error_codes,
    _get_bin_op,
    _get_select_op,
    _build_range,
)
from . import binaryop, unaryop, monoid, semiring as sring, types
from .scalar import Scalar
from .semiring import Semiring, current_semiring
from .binaryop import BinaryOp, current_accum, current_binop
from .unaryop import UnaryOp
from .monoid import Monoid, current_monoid
from . import descriptor
from .descriptor import Descriptor, Default, TransposeB

__all__ = ["Vector"]


class Vector:
    """GraphBLAS Sparse Vector

    This is a high-level wrapper around the low-level GrB_Vector type.

    """

    __slots__ = ("vector", "type", "_keep_alives")

    def _check(self, res, raise_no_val=False):
        if res != lib.GrB_SUCCESS:
            if raise_no_val and res == lib.GrB_NO_VALUE:
                raise KeyError

            error_string = ffi.new("char**")
            lib.GrB_Vector_error(error_string, self.vector[0])
            raise _error_codes[res](ffi.string(error_string[0]))

    def __init__(self, vec, typ=None):
        if typ is None:
            new_type = ffi.new("GrB_Type*")
            self._check(lib.GxB_Vector_type(new_type, vec[0]))

            typ = types.gb_type_to_type(new_type[0])

        self.vector = vec
        self.type = typ
        self._keep_alives = weakref.WeakKeyDictionary()

    def __del__(self):
        self._check(lib.GrB_Vector_free(self.vector))

    def __len__(self):
        return self.nvals

    def __iter__(self):
        nvals = self.nvals
        _nvals = ffi.new("GrB_Index[1]", [nvals])
        I = ffi.new("GrB_Index[%s]" % nvals)
        X = ffi.new("%s[%s]" % (self.type.C, nvals))
        self._check(self.type.Vector_extractTuples(I, X, _nvals, self.vector[0]))
        return zip(I, X)

    def iseq(self, other, eq_op=None):
        if eq_op is None:
            eq_op = self.type.EQ.get_binaryop(self.type, other.type)
        result = ffi.new("_Bool*")
        self._check(
            lib.LAGraph_Vector_isequal(result, self.vector[0], other.vector[0], eq_op)
        )
        return result[0]

    def isne(self, other):
        return not self.iseq(other)

    @classmethod
    def sparse(cls, typ, size=0):
        """Create an empty Vector from the given type and size."""
        new_vec = ffi.new("GrB_Vector*")
        _check(lib.GrB_Vector_new(new_vec, typ.gb_type, size))
        return cls(new_vec, typ)

    @classmethod
    def from_lists(cls, I, V, size=None, typ=None):
        """Create a new vector from the given lists of indices and values.  If
        size is not provided, it is computed from the max values of
        the provides size indices.

        """
        assert len(I) == len(V)
        assert len(I) > 0  # must be non empty
        if not size:
            size = max(I) + 1
        # TODO option to use ffi and GrB_Vector_build
        if typ is None:
            typ = types._gb_from_type(type(V[0]))
        m = cls.sparse(typ, size)
        for i, v in zip(I, V):
            m[i] = v
        return m

    @classmethod
    def from_list(cls, I):
        """Create a new dense vector from the given lists of values."""
        size = len(I)
        assert size > 0
        # TODO use ffi and GrB_Vector_build
        m = cls.sparse(types._gb_from_type(type(I[0])), size)
        for i, v in enumerate(I):
            m[i] = v
        return m

    @classmethod
    def from_1_to_n(cls, n):
        new_vec = ffi.new("GrB_Vector*")
        _check(lib.LAGraph_1_to_n(new_vec, n))
        if n < lib.INT32_MAX:
            return cls(new_vec, types.INT32)
        return cls(new_vec, types.INT64)  # pragma: no cover

    def dup(self):
        """Create an duplicate Vector from the given argument."""
        new_vec = ffi.new("GrB_Vector*")
        self._check(lib.GrB_Vector_dup(new_vec, self.vector[0]))
        return self.__class__(new_vec, self.type)

    @classmethod
    def dense(cls, typ, size, fill=None):
        v = cls.sparse(typ, size)
        if fill is None:
            fill = v.type.zero
        v[:] = fill
        return v

    def to_lists(self):
        """Extract the indices and values of the Vector as 2 lists."""
        I = ffi.new("GrB_Index[]", self.nvals)
        V = self.type.ffi.new(self.type.C + "[]", self.nvals)
        n = ffi.new("GrB_Index*")
        n[0] = self.nvals
        self._check(self.type.Vector_extractTuples(I, V, n, self.vector[0]))
        return [list(I), list(map(self.type.to_value, V))]

    def to_arrays(self):
        if self.type.typecode is None:
            raise TypeError("This matrix has no array typecode.")
        nvals = self.nvals
        _nvals = ffi.new("GrB_Index[1]", [nvals])
        I = ffi.new("GrB_Index[%s]" % nvals)
        X = self.type.ffi.new("%s[%s]" % (self.type.C, nvals))
        self._check(self.type.Vector_extractTuples(I, X, _nvals, self.vector[0]))
        return array("L", I), array(self.type.typecode, X)

    @property
    def size(self):
        """Return the size of the vector."""
        n = ffi.new("GrB_Index*")
        self._check(lib.GrB_Vector_size(n, self.vector[0]))
        return n[0]

    @property
    def shape(self):
        """Numpy-like description of vector shape."""
        return (self.size,)

    @property
    def nvals(self):
        """Return the number of values in the vector."""
        n = ffi.new("GrB_Index*")
        self._check(lib.GrB_Vector_nvals(n, self.vector[0]))
        return n[0]

    @property
    def gb_type(self):
        """Return the GraphBLAS low-level type object of the Vector."""
        typ = ffi.new("GrB_Type*")
        self._check(lib.GxB_Vector_type(typ, self.vector[0]))
        return typ[0]

    def full(self, identity=None):
        B = self.__class__.sparse(self.type, self.size)
        if identity is None:
            identity = self.type.one

        self._check(
            self.type.Vector_assignScalar(
                B.vector[0], NULL, NULL, identity, lib.GrB_ALL, 0, NULL
            )
        )
        return self.eadd(B, binaryop.FIRST)

    def compare(self, other, op, strop):
        C = self.__class__.sparse(types.BOOL, self.size)
        if isinstance(other, (bool, int, float, complex)):
            if op(other, 0):
                B = self.__class__.dup(self)
                B[:] = other
                self.emult(B, strop, out=C)
                return C
            else:
                self.select(strop, other).apply(lib.GxB_ONE_BOOL, out=C)
                return C
        elif isinstance(other, Vector):
            A = self.full()
            B = other.full()
            A.emult(B, strop, out=C)
            return C
        else:
            raise NotImplementedError

    def __gt__(self, other):
        return self.compare(other, operator.gt, ">")

    def __lt__(self, other):
        return self.compare(other, operator.lt, "<")

    def __ge__(self, other):
        return self.compare(other, operator.ge, ">=")

    def __le__(self, other):
        return self.compare(other, operator.le, "<=")

    def __eq__(self, other):
        return self.compare(other, operator.eq, "==")

    def __ne__(self, other):
        return self.compare(other, operator.ne, "!=")

    def eadd(self, other, add_op=NULL, cast=None, out=None, **kwargs):
        """Element-wise addition with other vector.

        Element-wise addition applies a binary operator element-wise
        on two vectors A and B, for all entries that appear in the set
        intersection of the patterns of A and B.  Other operators
        other than addition can be used.

        The pattern of the result of the element-wise addition is
        the set union of the pattern of A and B. Entries in neither in
        A nor in B do not appear in the result.

        """
        if add_op is NULL:
            add_op = current_binop.get(binaryop.PLUS)
        if isinstance(add_op, str):
            add_op = _get_bin_op(add_op, self.type)
        if isinstance(add_op, BinaryOp):
            add_op = add_op.get_binaryop(self.type, other.type)

        mask, mon, accum, desc = self._get_args(**kwargs)

        if out is None:
            typ = cast or types.promote(self.type, other.type)
            _out = ffi.new("GrB_Vector*")
            self._check(lib.GrB_Vector_new(_out, typ.gb_type, self.size))
            out = self.__class__(_out, typ)
        self._check(
            lib.GrB_Vector_eWiseAdd_BinaryOp(
                out.vector[0],
                mask,
                accum,
                add_op,
                self.vector[0],
                other.vector[0],
                desc.desc[0],
            )
        )
        return out

    def emult(
        self,
        other,
        mult_op=NULL,
        cast=None,
        out=None,
        **kwargs,
    ):
        """Element-wise multiplication with other vector.

        Element-wise multiplication applies a binary operator
        element-wise on two vectors A and B, for all entries that
        appear in the set intersection of the patterns of A and B.
        Other operators other than addition can be used.

        The pattern of the result of the element-wise multiplication
        is exactly this set intersection. Entries in A but not B, or
        visa versa, do not appear in the result.

        """
        if mult_op is NULL:
            mult_op = current_binop.get(binaryop.TIMES)
        if isinstance(mult_op, str):
            mult_op = _get_bin_op(mult_op, self.type)
        if isinstance(mult_op, BinaryOp):
            mult_op = mult_op.get_binaryop(self.type, other.type)
        mask, mon, accum, desc = self._get_args(**kwargs)
        if out is None:
            typ = cast or types.promote(self.type, other.type)
            _out = ffi.new("GrB_Vector*")
            self._check(lib.GrB_Vector_new(_out, typ.gb_type, self.size))
            out = self.__class__(_out, typ)
        self._check(
            lib.GrB_Vector_eWiseMult_BinaryOp(
                out.vector[0],
                mask,
                accum,
                mult_op,
                self.vector[0],
                other.vector[0],
                desc.desc[0],
            )
        )
        return out

    def vxm(self, other, cast=None, out=None, semiring=None, **kwargs):
        """Vector-Matrix multiply."""
        from .matrix import Matrix

        if semiring is None:
            semiring = current_semiring.get(None)

        mask, mon, accum, desc = self._get_args(**kwargs)
        typ = cast or types.promote(self.type, other.type, semiring)
        if out is None:
            new_dimension = other.nrows if TransposeB in desc else other.ncols
            out = Vector.sparse(typ, new_dimension)
        elif not isinstance(out, Vector):
            raise TypeError("Output argument must be Vector.")
        if semiring is None:
            semiring = typ.PLUS_TIMES
        self._check(
            lib.GrB_vxm(
                out.vector[0],
                mask,
                accum,
                semiring.get_semiring(typ),
                self.vector[0],
                other.matrix[0],
                desc.desc[0],
            )
        )
        return out

    def __matmul__(self, other):
        return self.vxm(other)

    def __imatmul__(self, other):
        return self.vxm(other, out=self)

    def __and__(self, other):
        mask, mon, accum, desc = self._get_args()
        return self.emult(other, mask=mask, accum=accum, desc=desc)

    def __iand__(self, other):
        mask, mon, accum, desc = self._get_args()
        return self.emult(other, mask=mask, accum=accum, desc=desc, out=self)

    def __or__(self, other):
        mask, mon, accum, desc = self._get_args()
        return self.eadd(other, mask=mask, accum=accum, desc=desc)

    def __ior__(self, other):
        mask, mon, accum, desc = self._get_args()
        return self.eadd(other, mask=mask, accum=accum, desc=desc, out=self)

    def __add__(self, other):
        mask, mon, accum, desc = self._get_args()
        if not isinstance(other, Vector):
            return self.apply_second(
                self.type.PLUS, other, mask=mask, accum=accum, desc=desc
            )
        return self.eadd(other, mask=mask, accum=accum, desc=desc)

    def __radd__(self, other):
        mask, mon, accum, desc = self._get_args()
        if not isinstance(other, Vector):
            return self.apply_first(
                other, self.type.PLUS, mask=mask, accum=accum, desc=desc
            )
        return other.eadd(self, mask=mask, accum=accum, desc=desc)

    def __iadd__(self, other):
        mask, mon, accum, desc = self._get_args()
        if not isinstance(other, Vector):
            return self.apply_second(
                self.type.PLUS, other, out=self, mask=mask, accum=accum, desc=desc
            )
        return self.eadd(other, out=self, mask=mask, accum=accum, desc=desc)

    def __sub__(self, other):
        mask, mon, accum, desc = self._get_args()
        if not isinstance(other, Vector):
            return self.apply_second(
                self.type.MINUS, other, mask=mask, accum=accum, desc=desc
            )
        return self.eadd(
            other, add_op=self.type.MINUS, mask=mask, accum=accum, desc=desc
        )

    def __rsub__(self, other):
        mask, mon, accum, desc = self._get_args()
        if not isinstance(other, Vector):
            return self.apply_first(
                other, self.type.MINUS, mask=mask, accum=accum, desc=desc
            )
        return other.eadd(
            self, add_op=self.type.MINUS, mask=mask, accum=accum, desc=desc
        )

    def __isub__(self, other):
        mask, mon, accum, desc = self._get_args()
        if not isinstance(other, Vector):
            return self.apply_second(
                self.type.MINUS, other, out=self, mask=mask, accum=accum, desc=desc
            )
        return other.eadd(
            self, out=self, add_op=self.type.MINUS, mask=mask, accum=accum, desc=desc
        )

    def __mul__(self, other):
        mask, mon, accum, desc = self._get_args()
        if not isinstance(other, Vector):
            return self.apply_second(
                self.type.TIMES, other, mask=mask, accum=accum, desc=desc
            )
        return self.eadd(
            other, add_op=self.type.TIMES, mask=mask, accum=accum, desc=desc
        )

    def __rmul__(self, other):
        mask, mon, accum, desc = self._get_args()
        if not isinstance(other, Vector):
            return self.apply_first(
                other, self.type.TIMES, mask=mask, accum=accum, desc=desc
            )
        return other.eadd(
            self, add_op=self.type.TIMES, mask=mask, accum=accum, desc=desc
        )

    def __imul__(self, other):
        mask, mon, accum, desc = self._get_args()
        if not isinstance(other, Vector):
            return self.apply_second(
                self.type.TIMES, other, out=self, mask=mask, accum=accum, desc=desc
            )
        return other.eadd(
            self, out=self, add_op=self.type.TIMES, mask=mask, accum=accum, desc=desc
        )

    def __truediv__(self, other):
        mask, mon, accum, desc = self._get_args()
        if not isinstance(other, Vector):
            return self.apply_second(
                self.type.DIV, other, mask=mask, accum=accum, desc=desc
            )
        return self.eadd(other, add_op=self.type.DIV, mask=mask, accum=accum, desc=desc)

    def __rtruediv__(self, other):
        mask, mon, accum, desc = self._get_args()
        if not isinstance(other, Vector):
            return self.apply_first(
                other, self.type.DIV, mask=mask, accum=accum, desc=desc
            )
        return other.eadd(self, add_op=self.type.DIV, mask=mask, accum=accum, desc=desc)

    def __itruediv__(self, other):
        mask, mon, accum, desc = self._get_args()
        if not isinstance(other, Vector):
            return self.apply_second(
                self.type.DIV, other, out=self, mask=mask, accum=accum, desc=desc
            )
        return other.eadd(
            self, out=self, add_op=self.type.DIV, mask=mask, accum=accum, desc=desc
        )

    def __invert__(self):
        return self.apply(unaryop.MINV)

    def __neg__(self):
        return self.apply(unaryop.AINV)

    def __abs__(self):
        return self.apply(unaryop.ABS)

    def clear(self):
        self._check(lib.GrB_Vector_clear(self.vector[0]))

    def resize(self, size):
        self._check(lib.GrB_Vector_resize(self.vector[0], size))

    def _get_args(self, mask=NULL, accum=NULL, mon=NULL, desc=Default):
        if mon is NULL:
            if self.type == types.BOOL:
                dmon = monoid.LOR_MONOID
            else:
                dmon = monoid.PLUS_MONOID
            mon = current_monoid.get(dmon)
        if isinstance(mon, Monoid):
            mon = mon.get_monoid(self.type)
        if accum is NULL:
            accum = current_accum.get(NULL)
        if isinstance(accum, BinaryOp):
            accum = accum.get_binaryop(self.type)
        if isinstance(mask, Vector):
            mask = mask.vector[0]
        return mask, mon, accum, desc

    def reduce_bool(self, mon=NULL, **kwargs):
        """Reduce vector to a boolean."""
        if mon is NULL:
            mon = current_monoid.get(types.BOOL.LOR_MONOID)
        mon = mon.get_monoid(self.type)
        mask, mon, accum, desc = self._get_args(mon=mon, **kwargs)
        result = ffi.new("_Bool*")
        self._check(
            lib.GrB_Vector_reduce_BOOL(result, accum, mon, self.vector[0], desc.desc[0])
        )
        return result[0]

    def reduce_int(self, mon=NULL, **kwargs):
        """Reduce vector to a integer."""
        if mon is NULL:
            mon = current_monoid.get(types.INT64.PLUS_MONOID)
        mon = mon.get_monoid(self.type)
        mask, mon, accum, desc = self._get_args(mon=mon, **kwargs)
        result = ffi.new("int64_t*")
        self._check(
            lib.GrB_Vector_reduce_INT64(
                result, accum, mon, self.vector[0], desc.desc[0]
            )
        )
        return result[0]

    def reduce_float(self, mon=NULL, **kwargs):
        """Reduce vector to a float."""
        if mon is NULL:
            mon = current_monoid.get(types.FP64.PLUS_MONOID)
        mon = mon.get_monoid(self.type)
        mask, mon, accum, desc = self._get_args(mon=mon, **kwargs)
        result = ffi.new("double*")
        self._check(
            lib.GrB_Vector_reduce_FP64(result, accum, mon, self.vector[0], desc.desc[0])
        )
        return result[0]

    def apply(self, op, out=None, **kwargs):
        """Apply Unary op to vector elements."""
        if out is None:
            out = Vector.sparse(self.type, self.size)
        if isinstance(op, UnaryOp):
            op = op.get_unaryop(self)

        mask, mon, accum, desc = self._get_args(**kwargs)
        self._check(
            lib.GrB_Vector_apply(
                out.vector[0], mask, accum, op, self.vector[0], desc.desc[0]
            )
        )
        return out

    def apply_first(self, first, op, out=None, **kwargs):
        """Apply a binary operator to the entries in a vector, binding the first input
        to a scalar first.
        """
        if out is None:
            out = self.__class__.sparse(self.type, self.size)
        if isinstance(op, BinaryOp):
            op = op.get_binaryop(self)
        mask, mon, accum, desc = self._get_args(**kwargs)
        if isinstance(first, Scalar):
            f = lib.GxB_Vector_apply_BinaryOp1st
            first = first.scalar[0]
        else:
            f = self.type.Vector_apply_BinaryOp1st
        self._check(
            f(out.vector[0], mask, accum, op, first, self.vector[0], desc.desc[0])
        )
        return out

    def apply_second(self, op, second, out=None, **kwargs):
        """Apply a binary operator to the entries in a vector, binding the second input
        to a scalar second.
        """
        if out is None:
            out = self.__class__.sparse(self.type, self.size)
        if isinstance(op, BinaryOp):
            op = op.get_binaryop(self)
        mask, mon, accum, desc = self._get_args(**kwargs)
        if isinstance(second, Scalar):
            f = lib.GxB_Vector_apply_BinaryOp2nd
            second = second.scalar[0]
        else:
            f = self.type.Vector_apply_BinaryOp2nd
        self._check(
            f(out.vector[0], mask, accum, op, self.vector[0], second, desc.desc[0])
        )
        return out

    def select(self, op, thunk=NULL, out=None, **kwargs):
        if out is None:
            out = Vector.sparse(self.type, self.size)
        if isinstance(op, str):
            op = _get_select_op(op)

        if isinstance(thunk, (bool, int, float, complex)):
            thunk = Scalar.from_value(thunk)
        if isinstance(thunk, Scalar):
            self._keep_alives[self.vector] = thunk
            thunk = thunk.scalar[0]

        mask, mon, accum, desc = self._get_args(**kwargs)
        self._check(
            lib.GxB_Vector_select(
                out.vector[0], mask, accum, op, self.vector[0], thunk, desc.desc[0]
            )
        )
        return out

    def pattern(self, typ=types.BOOL):
        """Return the pattern of the vector, this is a boolean Vector where
        every present value in this vector is set to True.
        """
        result = Vector.sparse(typ, self.size)
        self.apply(types.BOOL.ONE, out=result)
        return result

    def nonzero(self):
        return self.select(lib.GxB_NONZERO)

    def to_dense(self, _id=None):
        out = ffi.new("GrB_Vector*")
        if _id is None:
            _id = ffi.new(self.type.ptr, 0)
        self._check(lib.LAGraph_Vector_to_dense(out, self.vector[0], _id))
        return Vector(out, self.type)

    def __setitem__(self, index, value):
        mask, mon, accum, desc = self._get_args()
        if isinstance(index, int):
            val = self.type.from_value(value)
            self._check(self.type.Vector_setElement(self.vector[0], val, index))
            return

        if isinstance(index, slice):
            if isinstance(value, Vector):
                self.assign(value, index, mask=mask, desc=desc)
                return
            if isinstance(value, (bool, int, float, complex)):
                self.assign_scalar(value, index, mask=mask, desc=desc)
                return
        raise TypeError("Unknown index or value for vector assignment.")

    def assign(self, value, index=None, **kwargs):
        mask, mon, accum, desc = self._get_args(**kwargs)
        I, ni, size = _build_range(index, self.size - 1)
        self._check(
            lib.GrB_Vector_assign(
                self.vector[0], mask, accum, value.vector[0], I, ni, desc.desc[0]
            )
        )

    def assign_scalar(self, value, index=None, **kwargs):
        mask, mon, accum, desc = self._get_args(**kwargs)
        scalar_type = types._gb_from_type(type(value))
        I, ni, size = _build_range(index, self.size - 1)
        self._check(
            scalar_type.Vector_assignScalar(
                self.vector[0], mask, accum, value, I, ni, desc.desc[0]
            )
        )

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.extract_element(index)
        else:
            return self.extract(index)

    def __delitem__(self, index):
        if not isinstance(index, int):
            raise TypeError(
                "__delitem__ currently only supports single element removal"
            )
        self._check(lib.GrB_Vector_removeElement(self.vector[0], index))

    def extract_element(self, index):
        result = self.type.ffi.new(self.type.ptr)
        self._check(
            self.type.Vector_extractElement(
                result, self.vector[0], ffi.cast("GrB_Index", index)
            )
        )
        return self.type.to_value(result[0])

    def extract(self, index, **kwargs):
        mask, mon, accum, desc = self._get_args(**kwargs)
        if isinstance(index, Vector):
            mask = index.vector[0]
            index = slice(None, None, None)
        I, ni, size = _build_range(index, self.size - 1)
        if size is None:
            size = self.size
        result = Vector.sparse(self.type, size)
        self._check(
            lib.GrB_Vector_extract(
                result.vector[0], mask, accum, self.vector[0], I, ni, desc.desc[0]
            )
        )
        return result

    def __contains__(self, index):
        try:
            v = self[index]
            return True
        except NoValue:
            return False

    def get(self, i, default=None):
        try:
            return self[i]
        except NoValue:
            return default

    def wait(self):
        self._check(lib.GrB_Vector_wait(self.vector))

    def to_string(self, format_string="{:>%s}", width=2, empty_char=""):
        format_string = format_string % width
        result = ""
        for row in range(self.size):
            value = self.get(row, empty_char)
            result += str(row) + "|"
            result += format_string.format(self.type.format_value(value, width)) + "\n"
        return result

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return "<Vector (%s: %s:%s)>" % (self.size, self.nvals, self.type.__name__)
