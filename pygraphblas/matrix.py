import sys
import weakref
import operator
from random import randint
from array import array

from .base import (
    lib,
    ffi,
    NULL,
    NoValue,
    _check,
    _build_range,
    _get_select_op,
    _get_bin_op,
)

from . import types, binaryop, monoid, unaryop, semiring as _semiring
from .vector import Vector
from .scalar import Scalar
from .semiring import Semiring, current_semiring
from .binaryop import BinaryOp, current_accum, current_binop
from .unaryop import UnaryOp
from .monoid import Monoid, current_monoid
from . import descriptor
from .descriptor import Descriptor, Default, TransposeA, current_desc

__all__ = ["Matrix"]


class Matrix:
    """GraphBLAS Sparse Matrix

    This is a high-level wrapper around the GrB_Matrix type.

    """

    __slots__ = ("matrix", "type", "_funcs", "_keep_alives")

    def __init__(self, matrix, typ=None, **options):
        if typ is None:
            new_type = ffi.new("GrB_Type*")
            _check(lib.GxB_Matrix_type(new_type, matrix[0]))

            typ = types.gb_type_to_type(new_type[0])

        self.matrix = matrix
        self.type = typ
        self._keep_alives = weakref.WeakKeyDictionary()
        if options:
            self.options_set(**options)

    def __del__(self):
        _check(lib.GrB_Matrix_free(self.matrix))

    @classmethod
    def sparse(cls, typ, nrows=0, ncols=0, **options):
        """Create an empty Matrix from the given type, number of rows, and
        number of columns.

        """
        new_mat = ffi.new("GrB_Matrix*")
        _check(lib.GrB_Matrix_new(new_mat, typ.gb_type, nrows, ncols))
        m = cls(new_mat, typ, **options)
        return m

    @classmethod
    def dense(cls, typ, nrows=1, ncols=1, fill=None, **options):
        assert nrows > 0 and ncols > 0, "dense matrix must be at least 1x1"
        m = cls.sparse(typ, nrows, ncols, **options)
        if fill is None:
            fill = m.type.zero
        m[:, :] = fill
        return m

    @classmethod
    def from_lists(cls, I, J, V, nrows=None, ncols=None, typ=None, **options):
        """Create a new matrix from the given lists of row indices, column
        indices, and values.  If nrows or ncols are not provided, they
        are computed from the max values of the provides row and
        column indices lists.

        """
        assert len(I) == len(J) == len(V)
        if not nrows:
            nrows = max(I) + 1
        if not ncols:
            ncols = max(J) + 1
        # TODO use ffi and GrB_Matrix_build
        if typ is None:
            typ = types._gb_from_type(type(V[0]))
        m = cls.sparse(typ, nrows, ncols, **options)
        for i, j, v in zip(I, J, V):
            m[i, j] = v
        return m

    @classmethod
    def from_mm(cls, mm_file, typ, **options):
        """Create a new matrix by reading a Matrix Market file."""
        m = ffi.new("GrB_Matrix*")
        i = cls(m, typ, **options)
        _check(lib.LAGraph_mmread(m, mm_file))
        return i

    @classmethod
    def from_tsv(cls, tsv_file, typ, nrows, ncols, **options):
        """Create a new matrix by reading a tab separated value file."""
        m = ffi.new("GrB_Matrix*")
        i = cls(m, typ, **options)
        _check(lib.LAGraph_tsvread(m, tsv_file, typ.gb_type, nrows, ncols))
        return i

    @classmethod
    def from_binfile(cls, bin_file):
        """Create a new matrix by reading a SuiteSparse specific binary file."""
        m = ffi.new("GrB_Matrix*")
        _check(lib.LAGraph_binread(m, bin_file))
        return cls(m)

    @classmethod
    def random(
        cls,
        typ,
        nrows,
        ncols,
        nvals,
        make_pattern=False,
        make_symmetric=False,
        make_skew_symmetric=False,
        make_hermitian=True,
        no_diagonal=False,
        seed=None,
        **options
    ):
        """Create a new random Matrix of the given type, number of rows,
        columns and values.  Other flags set additional properties the
        matrix will hold.

        """
        result = ffi.new("GrB_Matrix*")
        i = cls(result, typ, **options)
        fseed = ffi.new("uint64_t*")
        if seed is None:
            seed = randint(0, sys.maxsize)
        fseed[0] = seed
        _check(
            lib.LAGraph_random(
                result,
                typ.gb_type,
                nrows,
                ncols,
                nvals,
                make_pattern,
                make_symmetric,
                make_skew_symmetric,
                make_hermitian,
                no_diagonal,
                fseed,
            )
        )
        return i

    @classmethod
    def identity(cls, typ, nrows, one=None, **options):
        result = cls.sparse(typ, nrows, nrows, **options)
        if one is None:
            one = result.type.one
        for i in range(nrows):
            result[i, i] = one
        return result

    @property
    def gb_type(self):
        """Return the GraphBLAS low-level type object of the Matrix."""
        new_type = ffi.new("GrB_Type*")
        _check(lib.GxB_Matrix_type(new_type, self.matrix[0]))
        return new_type[0]

    @property
    def nrows(self):
        """Return the number of Matrix rows."""
        n = ffi.new("GrB_Index*")
        _check(lib.GrB_Matrix_nrows(n, self.matrix[0]))
        return n[0]

    @property
    def ncols(self):
        """Return the number of Matrix columns."""
        n = ffi.new("GrB_Index*")
        _check(lib.GrB_Matrix_ncols(n, self.matrix[0]))
        return n[0]

    @property
    def shape(self):
        """Numpy-like description of matrix shape."""
        return (self.nrows, self.ncols)

    @property
    def square(self):
        return self.nrows == self.ncols

    @property
    def nvals(self):
        """Return the number of Matrix values."""
        n = ffi.new("GrB_Index*")
        _check(lib.GrB_Matrix_nvals(n, self.matrix[0]))
        return n[0]

    @property
    def T(self):
        return self.transpose()

    def dup(self, **options):
        """Create an duplicate Matrix."""
        new_mat = ffi.new("GrB_Matrix*")
        _check(lib.GrB_Matrix_dup(new_mat, self.matrix[0]))
        return self.__class__(new_mat, self.type, **options)

    def options_set(self, hyper=None, format=None):
        if hyper:
            hyper = ffi.cast("double", hyper)
            _check(lib.GxB_Matrix_Option_set(self.matrix[0], lib.GxB_HYPER, hyper))
        if format:
            format = ffi.cast("GxB_Format_Value", format)
            _check(lib.GxB_Matrix_Option_set(self.matrix[0], lib.GxB_FORMAT, format))

    def options_get(self):
        hyper = ffi.new("double*")
        _check(lib.GxB_Matrix_Option_get(self.matrix[0], lib.GxB_HYPER, hyper))

        format = ffi.new("GxB_Format_Value*")
        _check(lib.GxB_Matrix_Option_get(self.matrix[0], lib.GxB_FORMAT, format))

        is_hyper = ffi.new("bool*")
        _check(lib.GxB_Matrix_Option_get(self.matrix[0], lib.GxB_IS_HYPER, is_hyper))

        return (hyper[0], format[0], is_hyper[0])

    def pattern(self, typ=types.BOOL):
        """Return the pattern of the matrix, this is a boolean Matrix where
        every present value in this matrix is set to True.

        """

        r = ffi.new("GrB_Matrix*")
        _check(lib.LAGraph_pattern(r, self.matrix[0], typ.gb_type))
        return Matrix(r, typ)

    def to_mm(self, fileobj):
        """Write this matrix to a file using the Matrix Market format."""
        _check(lib.LAGraph_mmwrite(self.matrix[0], fileobj))

    def to_binfile(self, filename, comments=NULL):
        """Write this matrix using custom SuiteSparse binary format."""
        _check(lib.LAGraph_binwrite(self.matrix, filename, comments))

    def to_lists(self):
        """Extract the rows, columns and values of the Matrix as 3 lists."""
        I = ffi.new("GrB_Index[%s]" % self.nvals)
        J = ffi.new("GrB_Index[%s]" % self.nvals)
        V = self.type.ffi.new(self.type.C + "[%s]" % self.nvals)
        n = ffi.new("GrB_Index*")
        n[0] = self.nvals
        _check(self.type.Matrix_extractTuples(I, J, V, n, self.matrix[0]))
        return [list(I), list(J), list(map(self.type.to_value, V))]

    def clear(self):
        """Clear the matrix.  This does not change the size but removes all
        values.

        """
        _check(lib.GrB_Matrix_clear(self.matrix[0]))

    def resize(self, nrows, ncols):
        """Resize the matrix.  If the dimensions decrease, entries that fall
        outside the resized matrix are deleted.

        """
        _check(lib.GrB_Matrix_resize(self.matrix[0], nrows, ncols))

    def transpose(self, cast=None, out=None, **kwargs):
        """ Transpose matrix. """
        if out is None:
            new_dimensions = (
                (self.nrows, self.ncols)
                if TransposeA in kwargs.get("desc", ())
                else (self.ncols, self.nrows)
            )
            _out = ffi.new("GrB_Matrix*")
            if cast is not None:
                typ = cast
            else:
                typ = self.type
            _check(lib.GrB_Matrix_new(_out, typ.gb_type, *new_dimensions))
            out = self.__class__(_out, typ)
        mask, accum, desc = self._get_args(**kwargs)
        _check(lib.GrB_transpose(out.matrix[0], mask, accum, self.matrix[0], desc))
        return out

    def cast(self, cast, out=None):
        """Cast this matrix to the provided type.  If out is not provided, a
        new matrix is of the cast type is created.

        """
        return self.transpose(cast, out, desc=TransposeA)

    def eadd(self, other, add_op=NULL, cast=None, out=None, **kwargs):
        """Element-wise addition with other matrix.

        Element-wise addition applies a binary operator element-wise
        on two matrices A and B, for all entries that appear in the
        set intersection of the patterns of A and B.  Other operators
        other than addition can be used.

        The pattern of the result of the element-wise addition is
        the set union of the pattern of A and B. Entries in neither in
        A nor in B do not appear in the result.

        The only difference between element-wise multiplication and
        addition is the pattern of the result, and what happens to
        entries outside the intersection. With multiplication the
        pattern of T is the intersection; with addition it is the set
        union. Entries outside the set intersection are dropped for
        multiplication, and kept for addition; in both cases the
        operator is only applied to those (and only those) entries in
        the intersection. Any binary operator can be used
        interchangeably for either operation.

        """
        if add_op is NULL:
            add_op = current_binop.get(binaryop.PLUS)
        elif isinstance(add_op, str):
            add_op = _get_bin_op(add_op, self.type)

        add_op = add_op.get_binaryop(self.type, other.type)
        mask, accum, desc = self._get_args(**kwargs)
        if out is None:
            typ = cast or types.promote(self.type, other.type)
            _out = ffi.new("GrB_Matrix*")
            _check(lib.GrB_Matrix_new(_out, typ.gb_type, self.nrows, self.ncols))
            out = Matrix(_out, typ)

        _check(
            lib.GrB_eWiseAdd_Matrix_BinaryOp(
                out.matrix[0],
                mask,
                accum,
                add_op,
                self.matrix[0],
                other.matrix[0],
                desc,
            )
        )
        return out

    def emult(self, other, mult_op=NULL, cast=None, out=None, **kwargs):
        """Element-wise multiplication with other matrix.

        Element-wise multiplication applies a binary operator
        element-wise on two matrices A and B, for all entries that
        appear in the set intersection of the patterns of A and B.
        Other operators other than addition can be used.

        The pattern of the result of the element-wise multiplication
        is exactly this set intersection. Entries in A but not B, or
        visa versa, do not appear in the result.

        """
        if mult_op is NULL:
            mult_op = current_binop.get(binaryop.TIMES)
        elif isinstance(mult_op, str):
            mult_op = _get_bin_op(mult_op, self.type)

        mult_op = mult_op.get_binaryop(self.type, other.type)
        mask, accum, desc = self._get_args(**kwargs)
        if out is None:
            typ = cast or types.promote(self.type, other.type)
            _out = ffi.new("GrB_Matrix*")
            _check(lib.GrB_Matrix_new(_out, typ.gb_type, self.nrows, self.ncols))
            out = Matrix(_out, typ)

        _check(
            lib.GrB_eWiseMult_Matrix_BinaryOp(
                out.matrix[0],
                mask,
                accum,
                mult_op,
                self.matrix[0],
                other.matrix[0],
                desc,
            )
        )
        return out

    def iseq(self, other):
        """Compare two matrices for equality."""
        result = ffi.new("_Bool*")
        eq_op = self.type.EQ.get_binaryop(self.type, other.type)
        _check(lib.LAGraph_isequal(result, self.matrix[0], other.matrix[0], eq_op))
        return result[0]

    def isne(self, other):
        """Compare two matrices for inequality."""
        return not self.iseq(other)

    def __iter__(self):
        nvals = self.nvals
        _nvals = ffi.new("GrB_Index[1]", [nvals])
        I = ffi.new("GrB_Index[%s]" % nvals)
        J = ffi.new("GrB_Index[%s]" % nvals)
        X = self.type.ffi.new("%s[%s]" % (self.type.C, nvals))
        _check(self.type.Matrix_extractTuples(I, J, X, _nvals, self.matrix[0]))
        return zip(I, J, map(self.type.to_value, X))

    def to_arrays(self):
        if self.type.typecode is None:
            raise TypeError("This matrix has no array typecode.")
        nvals = self.nvals
        _nvals = ffi.new("GrB_Index[1]", [nvals])
        I = ffi.new("GrB_Index[%s]" % nvals)
        J = ffi.new("GrB_Index[%s]" % nvals)
        X = self.type.ffi.new("%s[%s]" % (self.type.C, nvals))
        _check(self.type.Matrix_extractTuples(I, J, X, _nvals, self.matrix[0]))
        return array("L", I), array("L", J), array(self.type.typecode, X)

    @property
    def rows(self):
        """An iterator of row indexes present in the matrix."""
        nvals = self.nvals
        _nvals = ffi.new("GrB_Index[1]", [nvals])
        I = ffi.new("GrB_Index[%s]" % nvals)
        J = NULL
        X = NULL
        _check(self.type.Matrix_extractTuples(I, J, X, _nvals, self.matrix[0]))
        return iter(I)

    @property
    def cols(self):
        """An iterator of column indexes present in the matrix."""
        nvals = self.nvals
        _nvals = ffi.new("GrB_Index[1]", [nvals])
        I = NULL
        J = ffi.new("GrB_Index[%s]" % nvals)
        X = NULL
        _check(self.type.Matrix_extractTuples(I, J, X, _nvals, self.matrix[0]))
        return iter(J)

    @property
    def vals(self):
        """An iterator of values present in the matrix."""
        nvals = self.nvals
        _nvals = ffi.new("GrB_Index[1]", [nvals])
        I = NULL
        J = NULL
        X = self.type.ffi.new("%s[%s]" % (self.type.C, nvals))
        _check(self.type.Matrix_extractTuples(I, J, X, _nvals, self.matrix[0]))
        return iter(X)

    def __len__(self):
        return self.nvals

    def __and__(self, other):
        mask, accum, desc = self._get_args()
        return self.emult(other, mask=mask, accum=accum, desc=desc)

    def __iand__(self, other):
        mask, accum, desc = self._get_args()
        return self.emult(other, mask=mask, accum=accum, desc=desc, out=self)

    def __or__(self, other):
        mask, accum, desc = self._get_args()
        return self.eadd(other, mask=mask, accum=accum, desc=desc)

    def __ior__(self, other):
        mask, accum, desc = self._get_args()
        return self.eadd(other, mask=mask, accum=accum, desc=desc, out=self)

    def __add__(self, other):
        mask, accum, desc = self._get_args()
        if not isinstance(other, Matrix):
            return self.apply_second(
                self.type.PLUS, other, mask=mask, accum=accum, desc=desc
            )
        return self.eadd(other, mask=mask, accum=accum, desc=desc)

    def __radd__(self, other):
        mask, accum, desc = self._get_args()
        if not isinstance(other, Matrix):
            return self.apply_first(
                other, self.type.PLUS, mask=mask, accum=accum, desc=desc
            )
        return other.eadd(self, mask=mask, accum=accum, desc=desc)

    def __iadd__(self, other):
        mask, accum, desc = self._get_args()
        if not isinstance(other, Matrix):
            return self.apply_second(
                self.type.PLUS, other, out=self, mask=mask, accum=accum, desc=desc
            )
        return self.eadd(other, out=self, mask=mask, accum=accum, desc=desc)

    def __sub__(self, other):
        mask, accum, desc = self._get_args()
        if not isinstance(other, Matrix):
            return self.apply_second(
                self.type.MINUS, other, mask=mask, accum=accum, desc=desc
            )
        return self.eadd(
            other, add_op=self.type.MINUS, mask=mask, accum=accum, desc=desc
        )

    def __rsub__(self, other):
        mask, accum, desc = self._get_args()
        if not isinstance(other, Matrix):
            return self.apply_first(
                other, self.type.MINUS, mask=mask, accum=accum, desc=desc
            )
        return other.eadd(
            self, add_op=self.type.MINUS, mask=mask, accum=accum, desc=desc
        )

    def __isub__(self, other):
        mask, accum, desc = self._get_args()
        if not isinstance(other, Matrix):
            return self.apply_second(
                self.type.MINUS, other, out=self, mask=mask, accum=accum, desc=desc
            )
        return other.eadd(
            self, out=self, add_op=self.type.MINUS, mask=mask, accum=accum, desc=desc
        )

    def __mul__(self, other):
        mask, accum, desc = self._get_args()
        if not isinstance(other, Matrix):
            return self.apply_second(
                self.type.TIMES, other, mask=mask, accum=accum, desc=desc
            )
        return self.eadd(
            other, add_op=self.type.TIMES, mask=mask, accum=accum, desc=desc
        )

    def __rmul__(self, other):
        mask, accum, desc = self._get_args()
        if not isinstance(other, Matrix):
            return self.apply_first(
                other, self.type.TIMES, mask=mask, accum=accum, desc=desc
            )
        return other.eadd(
            self, add_op=self.type.TIMES, mask=mask, accum=accum, desc=desc
        )

    def __imul__(self, other):
        mask, accum, desc = self._get_args()
        if not isinstance(other, Matrix):
            return self.apply_second(
                self.type.TIMES, other, out=self, mask=mask, accum=accum, desc=desc
            )
        return other.eadd(
            self, out=self, add_op=self.type.TIMES, mask=mask, accum=accum, desc=desc
        )

    def __truediv__(self, other):
        mask, accum, desc = self._get_args()
        if not isinstance(other, Matrix):
            return self.apply_second(
                self.type.DIV, other, mask=mask, accum=accum, desc=desc
            )
        return self.eadd(other, add_op=self.type.DIV, mask=mask, accum=accum, desc=desc)

    def __rtruediv__(self, other):
        mask, accum, desc = self._get_args()
        if not isinstance(other, Matrix):
            return self.apply_first(
                other, self.type.DIV, mask=mask, accum=accum, desc=desc
            )
        return other.eadd(self, add_op=self.type.DIV, mask=mask, accum=accum, desc=desc)

    def __itruediv__(self, other):
        mask, accum, desc = self._get_args()
        if not isinstance(other, Matrix):
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

    def __pow__(self, exponent):
        if exponent == 0:
            return self.__class__.identity(self.type, self.nrows)
        if exponent == 1:
            return self
        result = self.dup()
        for i in range(1, exponent):
            result.mxm(self, out=result)
        return result

    def reduce_bool(self, mon=NULL, **kwargs):
        """Reduce matrix to a boolean."""
        if mon is NULL:
            mon = current_monoid.get(types.BOOL.LOR_MONOID)
        mon = mon.get_monoid(self.type)
        result = ffi.new("_Bool*")
        mask, accum, desc = self._get_args(**kwargs)
        _check(lib.GrB_Matrix_reduce_BOOL(result, accum, mon, self.matrix[0], desc))
        return result[0]

    def reduce_int(self, mon=NULL, **kwargs):
        """Reduce matrix to an integer."""
        if mon is NULL:
            mon = current_monoid.get(types.INT64.PLUS_MONOID)
        mon = mon.get_monoid(self.type)
        result = ffi.new("int64_t*")
        mask, accum, desc = self._get_args(**kwargs)
        _check(lib.GrB_Matrix_reduce_INT64(result, accum, mon, self.matrix[0], desc))
        return result[0]

    def reduce_float(self, mon=NULL, **kwargs):
        """Reduce matrix to an float."""
        if mon is NULL:
            mon = current_monoid.get(self.type.PLUS_MONOID)
        mon = mon.get_monoid(self.type)
        mask, accum, desc = self._get_args(**kwargs)
        result = ffi.new("double*")
        _check(lib.GrB_Matrix_reduce_FP64(result, accum, mon, self.matrix[0], desc))
        return result[0]

    def reduce_vector(self, mon=NULL, out=None, **kwargs):
        """Reduce matrix to a vector."""
        if mon is NULL:
            mon = current_monoid.get(getattr(self.type, "PLUS_MONOID", NULL))
        mon = mon.get_monoid(self.type)
        if out is None:
            out = Vector.sparse(self.type, self.nrows)
        mask, accum, desc = self._get_args(**kwargs)
        _check(
            lib.GrB_Matrix_reduce_Monoid(
                out.vector[0], mask, accum, mon, self.matrix[0], desc
            )
        )
        return out

    def apply(self, op, out=None, **kwargs):
        """Apply Unary op to matrix elements."""
        if out is None:
            out = self.__class__.sparse(self.type, self.nrows, self.ncols)
        if isinstance(op, UnaryOp):
            op = op.get_unaryop(self.type)
        mask, accum, desc = self._get_args(**kwargs)
        _check(
            lib.GrB_Matrix_apply(out.matrix[0], mask, accum, op, self.matrix[0], desc)
        )
        return out

    def apply_first(self, first, op, out=None, **kwargs):
        """Apply a binary operator to the entries in a matrix, binding the first input
        to a scalar first.
        """
        if out is None:
            out = self.__class__.sparse(self.type, self.nrows, self.ncols)
        if isinstance(op, BinaryOp):
            op = op.get_binaryop(self.type)
        mask, accum, desc = self._get_args(**kwargs)
        if isinstance(first, Scalar):
            f = lib.GxB_Matrix_apply_BinaryOp1st
        else:
            f = self.type.Matrix_apply_BinaryOp1st
        _check(f(out.matrix[0], mask, accum, op, first, self.matrix[0], desc))
        return out

    def apply_second(self, op, second, out=None, **kwargs):
        """Apply a binary operator to the entries in a matrix, binding the second input
        to a scalar second.
        """
        if out is None:
            out = self.__class__.sparse(self.type, self.nrows, self.ncols)
        if isinstance(op, BinaryOp):
            op = op.get_binaryop(self.type)
        mask, accum, desc = self._get_args(**kwargs)
        _check(
            self.type.Matrix_apply_BinaryOp2nd(
                out.matrix[0], mask, accum, op, self.matrix[0], second, desc
            )
        )
        return out

    def select(self, op, thunk=NULL, out=NULL, **kwargs):
        if out is NULL:
            out = self.__class__.sparse(self.type, self.nrows, self.ncols)
        if isinstance(op, str):
            op = _get_select_op(op)

        if isinstance(thunk, (bool, int, float, complex)):
            thunk = Scalar.from_value(thunk)
        if isinstance(thunk, Scalar):
            self._keep_alives[self.matrix] = thunk
            thunk = thunk.scalar[0]

        mask, accum, desc = self._get_args(**kwargs)

        _check(
            lib.GxB_Matrix_select(
                out.matrix[0], mask, accum, op, self.matrix[0], thunk, desc
            )
        )
        return out

    def tril(self, thunk=NULL):
        return self.select(lib.GxB_TRIL, thunk=thunk)

    def triu(self, thunk=NULL):
        return self.select(lib.GxB_TRIU, thunk=thunk)

    def diag(self, thunk=NULL):
        return self.select(lib.GxB_DIAG, thunk=thunk)

    def offdiag(self, thunk=NULL):
        return self.select(lib.GxB_OFFDIAG, thunk=thunk)

    def nonzero(self):
        return self.select(lib.GxB_NONZERO)

    def full(self, identity=None):
        B = self.__class__.sparse(self.type, self.nrows, self.ncols)
        if identity is None:
            identity = self.type.one

        _check(
            self.type.Matrix_assignScalar(
                B.matrix[0], NULL, NULL, identity, lib.GrB_ALL, 0, lib.GrB_ALL, 0, NULL
            )
        )
        return self.eadd(B, self.type.FIRST)

    def compare(self, other, op, strop):
        C = self.__class__.sparse(types.BOOL, self.nrows, self.ncols)
        if isinstance(other, (bool, int, float, complex)):
            if op(other, 0):
                B = self.__class__.dup(self)
                B[:, :] = other
                self.emult(B, strop, out=C)
                return C
            else:
                self.select(strop, other).apply(types.BOOL.ONE, out=C)
                return C
        elif isinstance(other, Matrix):
            A = self.full()
            B = other.full()
            A.emult(B, strop, out=C)
            return C
        else:
            raise TypeError("Unknown matrix comparison type.")

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

    def _get_args(self, mask=NULL, accum=NULL, desc=NULL):
        if isinstance(mask, Matrix):
            mask = mask.matrix[0]
        elif isinstance(mask, Vector):
            mask = mask.vector[0]
        if accum is NULL:
            accum = current_accum.get(NULL)
        if isinstance(accum, BinaryOp):
            accum = accum.get_binaryop(self.type)
        if desc is NULL:
            desc = current_desc.get(Default)
        if isinstance(desc, Descriptor):
            desc = desc.desc[0]
        return mask, accum, desc

    def mxm(self, other, cast=None, out=None, semiring=None, **kwargs):
        """Matrix-matrix multiply."""
        if semiring is None:
            semiring = current_semiring.get(None)

        mask, accum, desc = self._get_args(**kwargs)
        typ = cast or types.promote(self.type, other.type, semiring)
        if out is None:
            out = self.__class__.sparse(typ, self.nrows, other.ncols)

        if semiring is None:
            semiring = typ.PLUS_TIMES

        _check(
            lib.GrB_mxm(
                out.matrix[0],
                mask,
                accum,
                semiring.get_semiring(typ),
                self.matrix[0],
                other.matrix[0],
                desc,
            )
        )
        return out

    def mxv(self, other, cast=None, out=None, semiring=None, **kwargs):
        """Matrix-vector multiply."""
        if semiring is None:
            semiring = current_semiring.get(None)

        mask, accum, desc = self._get_args(**kwargs)
        typ = cast or types.promote(self.type, other.type, semiring)
        if out is None:
            new_dimension = (
                self.ncols if TransposeA in kwargs.get("desc", ()) else self.nrows
            )
            out = Vector.sparse(typ, new_dimension)

        if semiring is None:
            semiring = typ.PLUS_TIMES

        _check(
            lib.GrB_mxv(
                out.vector[0],
                mask,
                accum,
                semiring.get_semiring(typ),
                self.matrix[0],
                other.vector[0],
                desc,
            )
        )
        return out

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            return self.mxm(other)
        elif isinstance(other, Vector):
            return self.mxv(other)
        else:
            raise TypeError("Right argument to @ must be Matrix or Vector.")

    def __imatmul__(self, other):
        return self.mxm(other, out=self)

    def kronecker(self, other, op=NULL, cast=None, out=None, **kwargs):
        """Kronecker product."""
        mask, accum, desc = self._get_args(**kwargs)
        typ = cast or types.promote(self.type, other.type)
        if out is None:
            out = self.__class__.sparse(
                typ, self.nrows * other.nrows, self.ncols * other.ncols
            )
        if op is NULL:
            op = typ.TIMES
        if isinstance(op, BinaryOp):
            op = op.get_binaryop(self.type, other.type)

        _check(
            lib.GrB_Matrix_kronecker_BinaryOp(
                out.matrix[0], mask, accum, op, self.matrix[0], other.matrix[0], desc
            )
        )
        return out

    kron = kronecker  # new v1.3 name

    def extract_matrix(self, rindex=None, cindex=None, out=None, **kwargs):
        """Slice a submatrix."""
        ta = TransposeA in kwargs.get("desc", ())
        mask, accum, desc = self._get_args(**kwargs)
        result_nrows = self.ncols if ta else self.nrows
        result_ncols = self.nrows if ta else self.ncols

        I, ni, isize = _build_range(rindex, result_nrows - 1)
        J, nj, jsize = _build_range(cindex, result_ncols - 1)
        if isize is None:
            isize = result_nrows
        if jsize is None:
            jsize = result_ncols

        if out is None:
            out = self.__class__.sparse(self.type, isize, jsize)

        _check(
            lib.GrB_Matrix_extract(
                out.matrix[0], mask, accum, self.matrix[0], I, ni, J, nj, desc
            )
        )
        return out

    def extract_col(self, col_index, row_slice=None, out=None, **kwargs):
        """Slice a column as subvector.
        Use `desc=TransposeA` to slice a row.
        """
        stop_val = self.ncols if TransposeA in kwargs.get("desc", ()) else self.nrows
        if out is None:
            out = Vector.sparse(self.type, stop_val)

        mask, accum, desc = self._get_args(**kwargs)
        I, ni, size = _build_range(row_slice, stop_val)

        _check(
            lib.GrB_Col_extract(
                out.vector[0], mask, accum, self.matrix[0], I, ni, col_index, desc
            )
        )
        return out

    def extract_row(self, row_index, col_slice=None, out=None, **kwargs):
        """Slice a row as subvector."""
        desc = TransposeA
        if "desc" in kwargs:
            desc = desc | kwargs["desc"]
        return self.extract_col(row_index, col_slice, out, desc=desc, **kwargs)

    def __getitem__(self, index):
        if isinstance(index, int):
            # a[3] extract single row
            return self.extract_row(index, None)
        if isinstance(index, slice):
            # a[3:] extract submatrix of rows
            return self.extract_matrix(index, None)

        if isinstance(index, Matrix):
            return self.extract_matrix(mask=index)

        if not isinstance(index, (tuple, list)):
            raise TypeError

        i0 = index[0]
        i1 = index[1]
        if isinstance(i0, int) and isinstance(i1, int):
            # a[3,3] extract single element
            result = self.type.ffi.new(self.type.ptr)
            _check(
                self.type.Matrix_extractElement(
                    result, self.matrix[0], index[0], index[1]
                )
            )
            return self.type.to_value(result[0])

        if isinstance(i0, int) and isinstance(i1, slice):
            # a[3,:] extract slice of row vector
            return self.extract_row(i0, i1)

        if isinstance(i0, slice) and isinstance(i1, int):
            # a[:,3] extract slice of col vector
            return self.extract_col(i1, i0)

        # a[:,:] or a[[0,1,2], [3,4,5]] extract submatrix with slice or row/col indices
        return self.extract_matrix(i0, i1)

    def assign_col(self, col_index, value, row_slice=None, **kwargs):
        """Assign a vector to a column."""
        stop_val = self.ncols if TransposeA in kwargs.get("desc", ()) else self.nrows
        I, ni, size = _build_range(row_slice, stop_val)
        mask, accum, desc = self._get_args(**kwargs)

        _check(
            lib.GrB_Col_assign(
                self.matrix[0], mask, accum, value.vector[0], I, ni, col_index, desc
            )
        )

    def assign_row(self, row_index, value, col_slice=None, **kwargs):
        """Assign a vector to a row."""
        stop_val = self.nrows if TransposeA in kwargs.get("desc", ()) else self.ncols
        I, ni, size = _build_range(col_slice, stop_val)

        mask, accum, desc = self._get_args(**kwargs)
        _check(
            lib.GrB_Row_assign(
                self.matrix[0], mask, accum, value.vector[0], row_index, I, ni, desc
            )
        )

    def assign_matrix(self, value, rindex=None, cindex=None, **kwargs):
        """Assign a submatrix."""
        I, ni, isize = _build_range(rindex, self.nrows - 1)
        J, nj, jsize = _build_range(cindex, self.ncols - 1)
        if isize is None:
            isize = self.nrows
        if jsize is None:
            jsize = self.ncols

        mask, accum, desc = self._get_args(**kwargs)

        _check(
            lib.GrB_Matrix_assign(
                self.matrix[0], mask, accum, value.matrix[0], I, ni, J, nj, desc
            )
        )

    def assign_scalar(self, value, row_slice=None, col_slice=None, **kwargs):
        mask, accum, desc = self._get_args(**kwargs)
        if row_slice:
            I, ni, isize = _build_range(row_slice, self.nrows - 1)
        else:
            I = lib.GrB_ALL
            ni = 0
        if col_slice:
            J, nj, jsize = _build_range(col_slice, self.ncols - 1)
        else:
            J = lib.GrB_ALL
            nj = 0
        scalar_type = types._gb_from_type(type(value))
        _check(
            scalar_type.Matrix_assignScalar(
                self.matrix[0], mask, accum, value, I, ni, J, nj, desc
            )
        )

    def __setitem__(self, index, value):
        if isinstance(index, int):
            # A[3] = assign single row  vector
            if isinstance(value, Vector):
                return self.assign_row(index, value)

        if isinstance(index, slice):
            # A[3:] = assign submatrix to rows
            if isinstance(value, Matrix):
                self.assign_matrix(value, index, None)
                return
            if isinstance(value, (bool, int, float, complex)):
                self.assign_scalar(value, index, None)
                return

        if isinstance(index, Matrix):
            if isinstance(value, Matrix):
                # A[M] = B masked matrix assignment
                raise NotImplementedError
            if not isinstance(value, (bool, int, float, complex)):
                raise TypeError
            # A[M] = s masked scalar assignment
            self.assign_scalar(value, mask=index)
            return

        if not isinstance(index, (tuple, list)):
            raise TypeError

        i0 = index[0]
        i1 = index[1]
        if isinstance(i0, int) and isinstance(i1, int):
            val = self.type.from_value(value)
            _check(self.type.Matrix_setElement(self.matrix[0], val, i0, i1))
            return

        if isinstance(i0, int) and isinstance(i1, slice):
            # a[3,:] assign slice of row vector or scalar
            self.assign_row(i0, value, i1)
            return

        if isinstance(i0, slice) and isinstance(i1, int):
            # a[:,3] extract slice of col vector or scalar
            self.assign_col(i1, value, i0)
            return

        if isinstance(i0, slice) and isinstance(i1, slice):
            if isinstance(value, (bool, int, float, complex)):
                self.assign_scalar(value, i0, i1)
                return

            # a[:,:] assign submatrix
            self.assign_matrix(value, i0, i1)
            return

    def __delitem__(self, index):
        if (
            not isinstance(index, tuple)
            or not isinstance(index[0], int)
            or not isinstance(index[1], int)
        ):
            raise TypeError(
                "__delitem__ currently only supports single element removal"
            )
        _check(lib.GrB_Matrix_removeElement(self.matrix[0], index[0], index[1]))

    def __contains__(self, index):
        try:
            v = self[index]
            return True
        except NoValue:
            return False

    def get(self, i, j, default=None):
        try:
            return self[i, j]
        except NoValue:
            return default

    def wait(self):
        _check(lib.GrB_Matrix_wait(self.matrix))

    def to_string(self, format_string="{:>%s}", width=3, empty_char=""):
        format_string = format_string % width
        header = (
            format_string.format("")
            + " "
            + "".join(format_string.format(i) for i in range(self.ncols))
        )
        result = header + "\n"
        for row in range(self.nrows):
            result += format_string.format(row) + "|"
            for col in range(self.ncols):
                value = self.get(row, col, empty_char)
                result += self.type.format_value(value, width)
            result += "|" + format_string.format(row) + "\n"
        result += header

        return result

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return "<Matrix (%sx%s : %s:%s)>" % (
            self.nrows,
            self.ncols,
            self.nvals,
            self.type.__name__,
        )
