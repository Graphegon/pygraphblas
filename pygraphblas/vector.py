"""High level wrapper around GraphBLAS Vectors.

"""
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
    GxB_INDEX_MAX,
)
from . import binaryop, unaryop, monoid, semiring as sring, types
from .scalar import Scalar
from .semiring import Semiring, current_semiring
from .binaryop import BinaryOp, current_accum, current_binop
from .unaryop import UnaryOp
from .monoid import Monoid, current_monoid
from . import descriptor
from .descriptor import Descriptor, Default, T1

__all__ = ["Vector"]
__pdoc__ = {"Vector.__init__": False}


class Vector:
    """GraphBLAS Sparse Vector

    This is a high-level wrapper around the low-level GrB_Vector type.

    """

    __slots__ = ("_vector", "type", "_keep_alives")

    def _check(self, res, raise_no_val=False):
        if res != lib.GrB_SUCCESS:
            if raise_no_val and res == lib.GrB_NO_VALUE:
                raise KeyError

            error_string = ffi.new("char**")
            lib.GrB_Vector_error(error_string, self._vector[0])
            raise _error_codes[res](ffi.string(error_string[0]))

    def __init__(self, vec, typ=None):
        if typ is None:
            new_type = ffi.new("GrB_Type*")
            self._check(lib.GxB_Vector_type(new_type, vec[0]))

            typ = types.gb_type_to_type(new_type[0])

        self._vector = vec
        self.type = typ
        self._keep_alives = weakref.WeakKeyDictionary()

    def __del__(self):
        self._check(lib.GrB_Vector_free(self._vector))

    def __len__(self):
        return self.nvals

    def __iter__(self):
        nvals = self.nvals
        _nvals = ffi.new("GrB_Index[1]", [nvals])
        I = ffi.new("GrB_Index[%s]" % nvals)
        X = ffi.new("%s[%s]" % (self.type.C, nvals))
        self._check(self.type.Vector_extractTuples(I, X, _nvals, self._vector[0]))
        return zip(I, X)

    def iseq(self, other, eq_op=None):
        """Compare two vectors for equality."""
        if eq_op is None:
            eq_op = self.type.EQ.get_binaryop(self.type, other.type)
        result = ffi.new("_Bool*")
        self._check(
            lib.LAGraph_Vector_isequal(result, self._vector[0], other._vector[0], eq_op)
        )
        return result[0]

    def isne(self, other):
        """Compare two vectors for inequality."""
        return not self.iseq(other)

    @classmethod
    def sparse(cls, typ, size=None):
        """Create an empty Vector from the given type.  If `size` is not
        specified it defaults to `pygraphblas.GxB_INDEX_MAX`.

        """
        if size is None:
            size = GxB_INDEX_MAX
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
        """Wrapper around LAGraph_1_to_n()"""
        new_vec = ffi.new("GrB_Vector*")
        _check(lib.LAGraph_1_to_n(new_vec, n))
        if n < lib.INT32_MAX:
            return cls(new_vec, types.INT32)
        return cls(new_vec, types.INT64)  # pragma: no cover

    def dup(self):
        """Create an duplicate Vector from the given argument."""
        new_vec = ffi.new("GrB_Vector*")
        self._check(lib.GrB_Vector_dup(new_vec, self._vector[0]))
        return self.__class__(new_vec, self.type)

    @classmethod
    def dense(cls, typ, size, fill=None):
        """Return a dense vector of `typ` and `size`.  If `fill` is provided,
        use that value otherwise use `type.zero`

        """
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
        self._check(self.type.Vector_extractTuples(I, V, n, self._vector[0]))
        return [list(I), list(map(self.type.to_value, V))]

    def to_arrays(self):
        """Return as python `array` objects."""
        if self.type.typecode is None:
            raise TypeError("This matrix has no array typecode.")
        nvals = self.nvals
        _nvals = ffi.new("GrB_Index[1]", [nvals])
        I = ffi.new("GrB_Index[%s]" % nvals)
        X = self.type.ffi.new("%s[%s]" % (self.type.C, nvals))
        self._check(self.type.Vector_extractTuples(I, X, _nvals, self._vector[0]))
        return array("L", I), array(self.type.typecode, X)

    @property
    def size(self):
        """Return the size of the vector."""
        n = ffi.new("GrB_Index*")
        self._check(lib.GrB_Vector_size(n, self._vector[0]))
        return n[0]

    @property
    def nvals(self):
        """Return the number of values in the vector."""
        n = ffi.new("GrB_Index*")
        self._check(lib.GrB_Vector_nvals(n, self._vector[0]))
        return n[0]

    @property
    def gb_type(self):
        """Return the GraphBLAS low-level type object of the Vector."""
        typ = ffi.new("GrB_Type*")
        self._check(lib.GxB_Vector_type(typ, self._vector[0]))
        return typ[0]

    def _full(self, identity=None):
        B = self.__class__.sparse(self.type, self.size)
        if identity is None:
            identity = self.type.one

        self._check(
            self.type.Vector_assignScalar(
                B._vector[0], NULL, NULL, identity, lib.GrB_ALL, 0, NULL
            )
        )
        return self.eadd(B, binaryop.FIRST)

    def _compare(self, other, op, strop):
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
            A = self._full()
            B = other._full()
            A.emult(B, strop, out=C)
            return C
        else:
            raise NotImplementedError

    def __gt__(self, other):
        return self._compare(other, operator.gt, ">")

    def __lt__(self, other):
        return self._compare(other, operator.lt, "<")

    def __ge__(self, other):
        return self._compare(other, operator.ge, ">=")

    def __le__(self, other):
        return self._compare(other, operator.le, "<=")

    def __eq__(self, other):
        return self._compare(other, operator.eq, "==")

    def __ne__(self, other):
        return self._compare(other, operator.ne, "!=")

    def eadd(
        self,
        other,
        add_op=None,
        cast=None,
        out=None,
        mask=None,
        accum=None,
        desc=Default,
    ):
        """Element-wise addition with other vector.

        Element-wise addition applies a binary operator element-wise
        on two vectors A and B, for all entries that appear in the set
        intersection of the patterns of A and B.  Other operators
        other than addition can be used.

        The pattern of the result of the element-wise addition is
        the set union of the pattern of A and B. Entries in neither in
        A nor in B do not appear in the result.

        """
        if add_op is None:
            add_op = current_binop.get(binaryop.PLUS)
        if isinstance(add_op, str):
            add_op = _get_bin_op(add_op, self.type)
        if isinstance(add_op, BinaryOp):
            add_op = add_op.get_binaryop(self.type, other.type)

        mask, accum, desc = self._get_args(mask, accum, desc)

        if out is None:
            typ = cast or types.promote(self.type, other.type)
            _out = ffi.new("GrB_Vector*")
            self._check(lib.GrB_Vector_new(_out, typ.gb_type, self.size))
            out = self.__class__(_out, typ)
        self._check(
            lib.GrB_Vector_eWiseAdd_BinaryOp(
                out._vector[0],
                mask,
                accum,
                add_op,
                self._vector[0],
                other._vector[0],
                desc.desc[0],
            )
        )
        return out

    def emult(
        self,
        other,
        mult_op=None,
        cast=None,
        out=None,
        mask=None,
        accum=None,
        desc=Default,
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
        if mult_op is None:
            mult_op = current_binop.get(binaryop.TIMES)
        if isinstance(mult_op, str):
            mult_op = _get_bin_op(mult_op, self.type)
        if isinstance(mult_op, BinaryOp):
            mult_op = mult_op.get_binaryop(self.type, other.type)
        mask, accum, desc = self._get_args(mask, accum, desc)
        if out is None:
            typ = cast or types.promote(self.type, other.type)
            _out = ffi.new("GrB_Vector*")
            self._check(lib.GrB_Vector_new(_out, typ.gb_type, self.size))
            out = self.__class__(_out, typ)
        self._check(
            lib.GrB_Vector_eWiseMult_BinaryOp(
                out._vector[0],
                mask,
                accum,
                mult_op,
                self._vector[0],
                other._vector[0],
                desc.desc[0],
            )
        )
        return out

    def vxm(
        self,
        other,
        cast=None,
        out=None,
        sring=None,
        mask=None,
        accum=None,
        desc=Default,
    ):
        """Vector-Matrix multiply."""

        if out is None:
            new_dimension = other.nrows if T1 in desc else other.ncols
            typ = cast or types.promote(self.type, other.type)
            out = Vector.sparse(typ, new_dimension)
        else:
            typ = out.type

        sring = current_semiring.get(typ.default_semiring())

        mask, accum, desc = self._get_args(mask, accum, desc)

        self._check(
            lib.GrB_vxm(
                out._vector[0],
                mask,
                accum,
                sring.get_semiring(typ),
                self._vector[0],
                other._matrix[0],
                desc.desc[0],
            )
        )
        return out

    def __matmul__(self, other):
        return self.vxm(other)

    def __imatmul__(self, other):
        return self.vxm(other, out=self)

    def __and__(self, other):
        return self.emult(other)

    def __iand__(self, other):
        return self.emult(other, out=self)

    def __or__(self, other):
        return self.eadd(other)

    def __ior__(self, other):
        return self.eadd(other, out=self)

    def __add__(self, other):
        if not isinstance(other, Vector):
            return self.apply_second(self.type.PLUS, other)
        return self.eadd(other)

    def __radd__(self, other):
        if not isinstance(other, Vector):
            return self.apply_first(other, self.type.PLUS)
        return other.eadd(self)

    def __iadd__(self, other):
        if not isinstance(other, Vector):
            return self.apply_second(self.type.PLUS, other, out=self)
        return self.eadd(other, out=self)

    def __sub__(self, other):
        if not isinstance(other, Vector):
            return self.apply_second(self.type.MINUS, other)
        return self.eadd(other, self.type.MINUS)

    def __rsub__(self, other):
        if not isinstance(other, Vector):
            return self.apply_first(other, self.type.MINUS)
        return other.eadd(self, self.type.MINUS)

    def __isub__(self, other):
        if not isinstance(other, Vector):
            return self.apply_second(self.type.MINUS, other)
        return other.eadd(self, self.type.MINUS, out=self)

    def __mul__(self, other):
        if not isinstance(other, Vector):
            return self.apply_second(self.type.TIMES, other)
        return self.eadd(other, self.type.TIMES)

    def __rmul__(self, other):
        if not isinstance(other, Vector):
            return self.apply_first(other, self.type.TIMES)
        return other.eadd(self, add_op=self.type.TIMES)

    def __imul__(self, other):
        if not isinstance(other, Vector):
            return self.apply_second(self.type.TIMES, other, out=self)
        return other.eadd(self, self.type.TIMES, out=self)

    def __truediv__(self, other):
        if not isinstance(other, Vector):
            return self.apply_second(self.type.DIV, other)
        return self.eadd(other, self.type.DIV)

    def __rtruediv__(self, other):
        if not isinstance(other, Vector):
            return self.apply_first(other, self.type.DIV)
        return other.eadd(self, self.type.DIV)

    def __itruediv__(self, other):
        if not isinstance(other, Vector):
            return self.apply_second(self.type.DIV, other, out=self)
        return other.eadd(self, self.type.DIV, out=self)

    def __invert__(self):
        return self.apply(unaryop.MINV)

    def __neg__(self):
        return self.apply(unaryop.AINV)

    def __abs__(self):
        return self.apply(unaryop.ABS)

    def clear(self):
        """Clear this vector removing all entries."""
        self._check(lib.GrB_Vector_clear(self._vector[0]))

    def resize(self, size):
        """Resize the vector.  If the dimensions decrease, entries that fall
        outside the resized vector are deleted.

        """
        self._check(lib.GrB_Vector_resize(self._vector[0], size))

    def _get_args(self, mask=None, accum=None, desc=Default):
        if accum is None or accum is NULL:
            accum = current_accum.get(NULL)
        if isinstance(accum, BinaryOp):
            accum = accum.get_binaryop(self.type)
        if mask is None:
            mask = NULL
        if isinstance(mask, Vector):
            mask = mask._vector[0]
        return mask, accum, desc

    def reduce_bool(self, mon=None, mask=None, accum=None, desc=Default):
        """Reduce vector to a boolean."""
        if mon is None:
            mon = current_monoid.get(types.BOOL.LOR_MONOID)
        mon = mon.get_monoid(self.type)
        mask, accum, desc = self._get_args(mask, accum, desc)
        result = ffi.new("_Bool*")
        self._check(
            lib.GrB_Vector_reduce_BOOL(
                result, accum, mon, self._vector[0], desc.desc[0]
            )
        )
        return result[0]

    def reduce_int(self, mon=None, mask=None, accum=None, desc=Default):
        """Reduce vector to a integer."""
        if mon is None:
            mon = current_monoid.get(types.INT64.PLUS_MONOID)
        mon = mon.get_monoid(self.type)
        mask, accum, desc = self._get_args(mask, accum, desc)
        result = ffi.new("int64_t*")
        self._check(
            lib.GrB_Vector_reduce_INT64(
                result, accum, mon, self._vector[0], desc.desc[0]
            )
        )
        return result[0]

    def reduce_float(self, mon=None, mask=None, accum=None, desc=Default):
        """Reduce vector to a float."""
        if mon is None:
            mon = current_monoid.get(types.FP64.PLUS_MONOID)
        mon = mon.get_monoid(self.type)
        mask, accum, desc = self._get_args(mask, accum, desc)
        result = ffi.new("double*")
        self._check(
            lib.GrB_Vector_reduce_FP64(
                result, accum, mon, self._vector[0], desc.desc[0]
            )
        )
        return result[0]

    def apply(self, op, out=None, mask=None, accum=None, desc=Default):
        """Apply Unary op to vector elements."""
        if out is None:
            out = Vector.sparse(self.type, self.size)
        if isinstance(op, UnaryOp):
            op = op.get_unaryop(self)

        mask, accum, desc = self._get_args(mask, accum, desc)
        self._check(
            lib.GrB_Vector_apply(
                out._vector[0], mask, accum, op, self._vector[0], desc.desc[0]
            )
        )
        return out

    def apply_first(self, first, op, out=None, mask=None, accum=None, desc=Default):
        """Apply a binary operator to the entries in a vector, binding the first input
        to a scalar first.
        """
        if out is None:
            out = self.__class__.sparse(self.type, self.size)
        if isinstance(op, BinaryOp):
            op = op.get_binaryop(self)
        mask, accum, desc = self._get_args(mask, accum, desc)
        if isinstance(first, Scalar):
            f = lib.GxB_Vector_apply_BinaryOp1st
            first = first.scalar[0]
        else:
            f = self.type.Vector_apply_BinaryOp1st
        self._check(
            f(out._vector[0], mask, accum, op, first, self._vector[0], desc.desc[0])
        )
        return out

    def apply_second(self, op, second, out=None, mask=None, accum=None, desc=Default):
        """Apply a binary operator to the entries in a vector, binding the second input
        to a scalar second.

        """
        if out is None:
            out = self.__class__.sparse(self.type, self.size)
        if isinstance(op, BinaryOp):
            op = op.get_binaryop(self)
        mask, accum, desc = self._get_args(mask, accum, desc)
        if isinstance(second, Scalar):
            f = lib.GxB_Vector_apply_BinaryOp2nd
            second = second.scalar[0]
        else:
            f = self.type.Vector_apply_BinaryOp2nd
        self._check(
            f(out._vector[0], mask, accum, op, self._vector[0], second, desc.desc[0])
        )
        return out

    def select(self, op, thunk=None, out=None, mask=None, accum=None, desc=Default):
        if out is None:
            out = Vector.sparse(self.type, self.size)
        if isinstance(op, str):
            op = _get_select_op(op)

        if thunk is None:
            thunk = NULL
        if isinstance(thunk, (bool, int, float, complex)):
            thunk = Scalar.from_value(thunk)
        if isinstance(thunk, Scalar):
            self._keep_alives[self._vector] = thunk
            thunk = thunk.scalar[0]

        mask, accum, desc = self._get_args(mask, accum, desc)
        self._check(
            lib.GxB_Vector_select(
                out._vector[0], mask, accum, op, self._vector[0], thunk, desc.desc[0]
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
        """Select vector of nonzero entries."""
        return self.select(lib.GxB_NONZERO)

    def to_dense(self, _id=None):
        """Convert to dense vector."""
        out = ffi.new("GrB_Vector*")
        if _id is None:
            _id = ffi.new(self.type.ptr, 0)
        self._check(lib.LAGraph_Vector_to_dense(out, self._vector[0], _id))
        return Vector(out, self.type)

    def __setitem__(self, index, value):
        if isinstance(index, int):
            val = self.type.from_value(value)
            self._check(self.type.Vector_setElement(self._vector[0], val, index))
            return

        if isinstance(index, slice):
            if isinstance(value, Vector):
                self.assign(value, index)
                return
            if isinstance(value, (bool, int, float, complex)):
                self.assign_scalar(value, index)
                return
        raise TypeError("Unknown index or value for vector assignment.")

    def assign(self, value, index=None, mask=None, accum=None, desc=Default):
        """Assign vector to vector."""
        mask, accum, desc = self._get_args(mask, accum, desc)
        I, ni, size = _build_range(index, self.size - 1)
        self._check(
            lib.GrB_Vector_assign(
                self._vector[0], mask, accum, value._vector[0], I, ni, desc.desc[0]
            )
        )

    def assign_scalar(self, value, index=None, mask=None, accum=None, desc=Default):
        """Assign scalar to vector."""
        mask, accum, desc = self._get_args(mask, accum, desc)
        scalar_type = types._gb_from_type(type(value))
        I, ni, size = _build_range(index, self.size - 1)
        self._check(
            scalar_type.Vector_assignScalar(
                self._vector[0], mask, accum, value, I, ni, desc.desc[0]
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
        self._check(lib.GrB_Vector_removeElement(self._vector[0], index))

    def extract_element(self, index):
        """Extract element from vector."""
        result = self.type.ffi.new(self.type.ptr)
        self._check(
            self.type.Vector_extractElement(
                result, self._vector[0], ffi.cast("GrB_Index", index)
            )
        )
        return self.type.to_value(result[0])

    def extract(self, index, mask=None, accum=None, desc=Default):
        """Extract subvector from vector."""
        mask, accum, desc = self._get_args(mask, accum, desc)
        if isinstance(index, Vector):
            mask = index._vector[0]
            index = slice(None, None, None)
        I, ni, size = _build_range(index, self.size - 1)
        if size is None:
            size = self.size
        result = Vector.sparse(self.type, size)
        self._check(
            lib.GrB_Vector_extract(
                result._vector[0], mask, accum, self._vector[0], I, ni, desc.desc[0]
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
        """Get element at `i` or return `default` if not present."""
        try:
            return self[i]
        except NoValue:
            return default

    def wait(self):
        """Wait for vector to complete."""
        self._check(lib.GrB_Vector_wait(self._vector))

    def to_string(self, format_string="{:>%s}", width=2, empty_char=""):
        """Return string representation of vector."""
        format_string = format_string % width
        result = ""
        for row in range(self.size):
            value = self.get(row, empty_char)
            result += str(row) + "|"
            result += format_string.format(
                self.type.format_value(value, width)
            ).rstrip()
            if row < self.size - 1:
                result += "\n"
        return result

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return "<Vector (%s: %s:%s)>" % (self.size, self.nvals, self.type.__name__)
