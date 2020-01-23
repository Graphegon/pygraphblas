import sys
import types as pytypes
import weakref
import operator
from random import randint

from .base import (
    lib,
    ffi,
    NULL,
    _check,
    _build_range,
    _get_select_op,
    _get_bin_op,
)

from . import types
from . import binaryop
from .vector import Vector
from .scalar import Scalar
from .semiring import Semiring, current_semiring
from .binaryop import BinaryOp, current_accum, current_binop
from .unaryop import UnaryOp
from . import descriptor

__all__ = ['Matrix']

class Matrix:
    """GraphBLAS Sparse Matrix

    This is a high-level wrapper around the GrB_Matrix type.

    """

    __slots__ = ('matrix', 'type', '_funcs', '_keep_alives')

    def __init__(self, matrix, typ):
        self.matrix = matrix
        self.type = typ
        self._keep_alives = weakref.WeakKeyDictionary()

    def __del__(self):
        _check(lib.GrB_Matrix_free(self.matrix))

    @classmethod
    def from_type(cls, typ, nrows=0, ncols=0):
        """Create an empty Matrix from the given type, number of rows, and
        number of columns.

        """
        new_mat = ffi.new('GrB_Matrix*')
        _check(lib.GrB_Matrix_new(new_mat, typ.gb_type, nrows, ncols))
        return cls(new_mat, typ)

    @classmethod
    def dense(cls, typ, nrows, ncols, fill=None):
        m = cls.from_type(typ, nrows, ncols)
        if fill is None:
            fill = m.type.aidentity
            m[:,:] = fill
        return m

    @classmethod
    def from_lists(cls, I, J, V, nrows=None, ncols=None):
        """Create a new matrix from the given lists of row indices, column
        indices, and values.  If nrows or ncols are not provided, they
        are computed from the max values of the provides row and
        column indices lists.

        """
        assert len(V)
        assert len(I) == len(J) == len(V)
        if not nrows:
            nrows = max(I) + 1
        if not ncols:
            ncols = max(J) + 1
        # TODO use ffi and GrB_Matrix_build
        typ = types._gb_from_type(type(V[0]))
        m = cls.from_type(typ, nrows, ncols)
        for i, j, v in zip(I, J, V):
            m[i, j] = v
        return m

    @classmethod
    def from_mm(cls, mm_file):
        """Create a new matrix by reading a Matrix Market file.

        """
        m = ffi.new('GrB_Matrix*')
        _check(lib.LAGraph_mmread(m, mm_file))
        return cls(m)

    @classmethod
    def from_tsv(cls, tsv_file, typ, nrows, ncols):
        """Create a new matrix by reading a Matrix Market file.

        """
        m = ffi.new('GrB_Matrix*')
        _check(lib.LAGraph_tsvread(m, tsv_file, typ.gb_type, nrows, ncols))
        return cls(m, typ)

    @classmethod
    def from_random(cls, typ, nrows, ncols, nvals,
                    make_pattern=False, make_symmetric=False,
                    make_skew_symmetric=False, make_hermitian=True,
                    no_diagonal=False, seed=None):
        """Create a new random Matrix of the given type, number of rows,
        columns and values.  Other flags set additional properties the
        matrix will hold.

        """
        result = ffi.new('GrB_Matrix*')
        fseed = ffi.new('uint64_t*')
        if seed is None:
            seed = randint(0, sys.maxsize)
        fseed[0] = seed
        _check(lib.LAGraph_random(
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
            fseed))
        return cls(result, typ)

    @classmethod
    def identity(cls, typ, nrows, ncols):
        result = cls.from_type(typ, nrows, ncols)
        for i in range(nrows):
            result[i,i] = result.type.aidentity
        return result

    @property
    def gb_type(self):
        """Return the GraphBLAS low-level type object of the Matrix.

        """
        new_type = ffi.new('GrB_Type*')
        _check(lib.GxB_Matrix_type(new_type, self.matrix[0]))
        return new_type[0]

    @property
    def nrows(self):
        """Return the number of Matrix rows.

        """
        n = ffi.new('GrB_Index*')
        _check(lib.GrB_Matrix_nrows(n, self.matrix[0]))
        return n[0]

    @property
    def ncols(self):
        """Return the number of Matrix columns.

        """
        n = ffi.new('GrB_Index*')
        _check(lib.GrB_Matrix_ncols(n, self.matrix[0]))
        return n[0]

    @property
    def shape(self):
        """Numpy-like description of matrix shape.

        """
        return (self.nrows, self.ncols)

    @property
    def nvals(self):
        """Return the number of Matrix values.

        """
        n = ffi.new('GrB_Index*')
        _check(lib.GrB_Matrix_nvals(n, self.matrix[0]))
        return n[0]

    def dup(self):
        """Create an duplicate Matrix.

        """
        new_mat = ffi.new('GrB_Matrix*')
        _check(lib.GrB_Matrix_dup(new_mat, self.matrix[0]))
        return self.__class__(new_mat, self.type)

    def options_set(self, hyper=None, format=None):
        if hyper:
            hyper = ffi.cast('double', hyper)
            _check(lib.GxB_Matrix_Option_set(
                self.matrix[0],
                lib.GxB_HYPER,
                hyper))
        if format:
            format = ffi.cast('GxB_Format_Value', format)
            _check(lib.GxB_Matrix_Option_set(
                self.matrix[0],
                lib.GxB_FORMAT,
                format))

    def options_get(self):
        hyper = ffi.new('double*')
        _check(lib.GxB_Matrix_Option_get(
            self.matrix[0],
            lib.GxB_HYPER,
            hyper
            ))

        format = ffi.new('GxB_Format_Value*')
        _check(lib.GxB_Matrix_Option_get(
            self.matrix[0],
            lib.GxB_FORMAT,
            format
            ))

        is_hyper = ffi.new('bool*')
        _check(lib.GxB_Matrix_Option_get(
            self.matrix[0],
            lib.GxB_IS_HYPER,
            is_hyper
            ))

        return (hyper[0], format[0], is_hyper[0])

    def pattern(self):
        """Return the pattern of the matrix, this is a boolean Matrix where
        every present value in this matrix is set to True.

        """

        r = ffi.new('GrB_Matrix*')
        _check(lib.LAGraph_pattern(r, self.matrix[0]))
        return Matrix(r, types.BOOL)

    def to_mm(self, fileobj):
        """Write this matrix to a file using the Matrix Market format.

        """
        _check(lib.LAGraph_mmwrite(self.matrix[0], fileobj))

    def to_lists(self):
        """Extract the rows, columns and values of the Matrix as 3 lists.

        """
        I = ffi.new('GrB_Index[%s]' % self.nvals)
        J = ffi.new('GrB_Index[%s]' % self.nvals)
        V = self.type.ffi.new(self.type.C + '[%s]' % self.nvals)
        n = ffi.new('GrB_Index*')
        n[0] = self.nvals
        _check(self.type.Matrix_extractTuples(
            I,
            J,
            V,
            n,
            self.matrix[0]
            ))
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
        _check(lib.GxB_Matrix_resize(
            self.matrix[0],
            nrows,
            ncols))

    def transpose(self, out=None, **kwargs):
        """ Transpose matrix. """
        if out is None:
            _out = ffi.new('GrB_Matrix*')
            _check(lib.GrB_Matrix_new(
                _out, self.type.gb_type, self.nrows, self.ncols))
            out = self.__class__(_out, self.type)
        mask, semiring, accum, desc = self._get_args(**kwargs)
        _check(lib.GrB_transpose(
            out.matrix[0],
            mask,
            accum,
            self.matrix[0],
            desc
            ))
        return out

    def eadd(self, other, add_op=NULL, out=None, **kwargs):
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
            add_op = current_binop.get(self.type.add_op)
        elif isinstance(add_op, str):
            add_op = _get_bin_op(add_op, self.type)
        if isinstance(add_op, BinaryOp):
            add_op = add_op.get_binaryop(self, other)
        if out is None:
            _out = ffi.new('GrB_Matrix*')
            _check(lib.GrB_Matrix_new(
                _out, self.type.gb_type, self.nrows, self.ncols))
            out = Matrix(_out, self.type)
        mask, semiring, accum, desc = self._get_args(**kwargs)

        _check(lib.GrB_eWiseAdd_Matrix_BinaryOp(
            out.matrix[0],
            mask,
            accum,
            add_op,
            self.matrix[0],
            other.matrix[0],
            desc))
        return out

    def emult(self, other, mult_op=NULL, out=None, **kwargs):
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
            mult_op = current_binop.get(self.type.mult_op)
        elif isinstance(mult_op, str):
            mult_op = _get_bin_op(mult_op, self.type)
        if isinstance(mult_op, BinaryOp):
            mult_op = mult_op.get_binaryop(self, other)
        if out is None:
            _out = ffi.new('GrB_Matrix*')
            _check(lib.GrB_Matrix_new(
                _out, self.type.gb_type, self.nrows, self.ncols))
            out = Matrix(_out, self.type)
        mask, semiring, accum, desc = self._get_args(**kwargs)

        _check(lib.GrB_eWiseMult_Matrix_BinaryOp(
            out.matrix[0],
            mask,
            accum,
            mult_op,
            self.matrix[0],
            other.matrix[0],
            desc))
        return out

    def iseq(self, other):
        """Compare two matrices for equality.
        """
        result = ffi.new('_Bool*')
        if isinstance(self.type.eq_op, BinaryOp):
            eq_op = self.type.eq_op.get_binaryop(self, other)
        else:
            eq_op = self.type.eq_op
        _check(lib.LAGraph_isequal(
            result,
            self.matrix[0],
            other.matrix[0],
            eq_op))
        return result[0]

    def isne(self, other):
        """Compare two matrices for inequality.
        """
        return not self.iseq(other)

    def __getstate__(self):
        pass

    def __setstate__(self, data):
        pass

    def __iter__(self):
        nvals = self.nvals
        _nvals = ffi.new('GrB_Index[1]', [nvals])
        I = ffi.new('GrB_Index[%s]' % nvals)
        J = ffi.new('GrB_Index[%s]' % nvals)
        X = self.type.ffi.new('%s[%s]' % (self.type.C, nvals))
        _check(self.type.Matrix_extractTuples(
            I,
            J,
            X,
            _nvals,
            self.matrix[0]
            ))
        return zip(I, J, map(self.type.to_value, X))


    def to_arrays(self):
        nvals = self.nvals
        _nvals = ffi.new('GrB_Index[1]', [nvals])
        I = ffi.new('GrB_Index[%s]' % nvals)
        J = ffi.new('GrB_Index[%s]' % nvals)
        X = self.type.ffi.new('%s[%s]' % (self.type.C, nvals))
        _check(self.type.Matrix_extractTuples(
            I,
            J,
            X,
            _nvals,
            self.matrix[0]
            ))
        

    @property
    def rows(self):
        """ An iterator of row indexes present in the matrix.
        """
        for i, j, v in self:
            yield i

    @property
    def cols(self):
        """ An iterator of column indexes present in the matrix.
        """
        for i, j, v in self:
            yield j

    @property
    def vals(self):
        """ An iterator of values present in the matrix.
        """
        for i, j, v in self:
            yield v

    def __len__(self):
        return self.nvals

    def __nonzero__(self):
        return self.reduce_bool()

    def __add__(self, other):
        return self.eadd(other)

    def __iadd__(self, other):
        return self.eadd(other, out=self)

    def __sub__(self, other):
        return self.eadd(other, add_op=binaryop.sub)

    def __isub__(self, other):
        return self.eadd(other, add_op=binaryop.sub, out=self)

    def __mul__(self, other):
        return self.emult(other)

    def __imul__(self, other):
        return self.emult(other, out=self)

    def __truediv__(self, other):
        return self.emult(other, mult_op=binaryop.div)

    def __itruediv__(self, other):
        return self.emult(other, mult_op=binaryop.div, out=self)

    def __invert__(self):
        return self.apply(self.type.invert)

    def __neg__(self):
        return self.apply(self.type.neg)

    def __abs__(self):
        return self.apply(self.type.abs_)

    def __pow__(self, exponent):
        if exponent == 0:
            return self.__class__.identity(self.type, self.nrows, self.ncols)
        if exponent == 1:
            return self
        result = self
        for i in range(1, exponent):
            result = result @ self
        return result

    def reduce_bool(self, monoid=NULL, **kwargs):
        """Reduce matrix to a boolean.

        """
        if monoid is NULL:
            monoid = lib.GxB_LOR_BOOL_MONOID
        result = ffi.new('_Bool*')
        mask, semiring, accum, desc = self._get_args(**kwargs)
        _check(lib.GrB_Matrix_reduce_BOOL(
            result,
            accum,
            monoid,
            self.matrix[0],
            desc))
        return result[0]

    def reduce_int(self, monoid=NULL, **kwargs):
        """Reduce matrix to an integer.

        """
        if monoid is NULL:
            monoid = lib.GxB_PLUS_INT64_MONOID
        result = ffi.new('int64_t*')
        mask, semiring, accum, desc = self._get_args(**kwargs)
        _check(lib.GrB_Matrix_reduce_INT64(
            result,
            accum,
            monoid,
            self.matrix[0],
            desc))
        return result[0]

    def reduce_float(self, monoid=NULL, **kwargs):
        """Reduce matrix to an float.

        """
        if monoid is NULL:
            monoid = lib.GxB_PLUS_FP64_MONOID
        mask, semiring, accum, desc = self._get_args(**kwargs)
        result = ffi.new('double*')
        _check(lib.GrB_Matrix_reduce_FP64(
            result,
            accum,
            monoid,
            self.matrix[0],
            desc))
        return result[0]

    def reduce_vector(self, out=None, monoid=NULL, **kwargs):
        """Reduce matrix to a vector.

        """
        if monoid is NULL:
            monoid = self.type.monoid
        if out is None:
            out = Vector.from_type(self.type, self.nrows)
        mask, semiring, accum, desc = self._get_args(**kwargs)
        _check(lib.GrB_Matrix_reduce_Monoid(
            out.vector[0],
            mask,
            accum,
            monoid,
            self.matrix[0],
            desc))
        return out

    def apply(self, op, out=None, **kwargs):
        """Apply Unary op to matrix elements.

        """
        if out is None:
            out = self.__class__.from_type(self.type, self.nrows, self.ncols)
        if isinstance(op, UnaryOp):
            nop = op.unaryop
        elif isinstance(op, pytypes.FunctionType):
            uop = ffi.new('GrB_UnaryOp*')
            def op_func(z, x):
                z = self.type.ffi.cast(self.type.ptr, z)
                x = self.type.ffi.cast(self.type.ptr, x)
                z[0] = op(x[0])
            func = self.type.ffi.callback('void(void*, const void*)', op_func)
            self._keep_alives[self.matrix] = (op, uop, func)
            _check(lib.GrB_UnaryOp_new(
                uop,
                func,
                self.type.gb_type,
                self.type.gb_type
                ))
            self._keep_alives[self.matrix] = (op, uop, func)
            nop = uop[0]
        else:
            nop = op

        mask, semiring, accum, desc = self._get_args(**kwargs)
        _check(lib.GrB_Matrix_apply(
            out.matrix[0],
            mask,
            accum,
            nop,
            self.matrix[0],
            desc
            ))
        return out

    def select(self, op, thunk=NULL, out=NULL, **kwargs):
        if out is NULL:
            out = self.__class__.from_type(self.type, self.nrows, self.ncols)
        if isinstance(op, UnaryOp):
            op = op.unaryop
        elif isinstance(op, str):
            op = _get_select_op(op)

        if isinstance(thunk, (bool, int, float)):
            thunk = Scalar.from_value(thunk)
        if isinstance(thunk, Scalar):
            self._keep_alives[self.matrix] = thunk
            thunk = thunk.scalar[0]

        mask, semiring, accum, desc = self._get_args(**kwargs)

        _check(lib.GxB_Matrix_select(
            out.matrix[0],
            mask,
            accum,
            op,
            self.matrix[0],
            thunk,
            desc
            ))
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
        B = self.__class__.from_type(self.type, self.nrows, self.ncols)
        if identity is None:
            identity = self.type.identity

        _check(self.type.Matrix_assignScalar(
            B.matrix[0],
            NULL,
            NULL,
            identity,
            lib.GrB_ALL,
            0,
            lib.GrB_ALL,
            0,
            NULL))
        return self.eadd(B, self.type.first)

    def compare(self, other, op, strop):
        C = self.__class__.from_type(types.BOOL, self.nrows, self.ncols)
        if isinstance(other, (bool, int, float)):
            if op(other, 0):
                B = self.__class__.dup(self)
                B[:,:] = other
                self.emult(B, strop, out=C)
                return C
            else:
                self.select(strop, other).apply(lib.GxB_ONE_BOOL, out=C)
                return C
        elif isinstance(other, Matrix):
            A = self.full()
            B = other.full()
            A.emult(B, strop, out=C)
            return C
        else:
            raise NotImplementedError

    def __gt__(self, other):
        return self.compare(other, operator.gt, '>')

    def __lt__(self, other):
        return self.compare(other, operator.lt, '<')

    def __ge__(self, other):
        return self.compare(other, operator.ge, '>=')

    def __le__(self, other):
        return self.compare(other, operator.le, '<=')

    def __eq__(self, other):
        return self.compare(other, operator.eq, '==')

    def __ne__(self, other):
        return self.compare(other, operator.ne, '!=')

    def _get_args(self,
            mask=NULL, accum=NULL, semiring=NULL, desc=descriptor.oooo):
        if isinstance(mask, Matrix):
            mask = mask.matrix[0]
        if isinstance(mask, Vector):
            mask = mask.vector[0]
        if semiring is NULL:
            semiring = current_semiring.get(self.type.semiring)
        if isinstance(semiring, Semiring):
            semiring = semiring.get_semiring(self)
        if accum is NULL:
            accum = current_accum.get(NULL)
        elif isinstance(accum, BinaryOp):
            accum = accum.get_binaryop(self)
        return mask, semiring, accum, desc

    def mxm(self, other, out=None, **kwargs):
        """Matrix-matrix multiply.

        """
        if out is None:
            out = self.__class__.from_type(self.type, self.nrows, other.ncols)

        mask, semiring, accum, desc = self._get_args(**kwargs)
        _check(lib.GrB_mxm(
            out.matrix[0],
            mask,
            accum,
            semiring,
            self.matrix[0],
            other.matrix[0],
            desc))
        return out

    def mxv(self, other, out=None, **kwargs):
        """Matrix-vector multiply.

        """
        if out is None:
            out = Vector.from_type(self.type, self.ncols)
        mask, semiring, accum, desc = self._get_args(**kwargs)
        _check(lib.GrB_mxv(
            out.vector[0],
            mask,
            accum,
            semiring,
            self.matrix[0],
            other.vector[0],
            desc))
        return out

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            return self.mxm(other)
        elif isinstance(other, Vector):
            return self.mxv(other)
        else:
            raise TypeError('Right argument to @ must be Matrix or Vector.')

    def __imatmul__(self, other):
        return self.mxm(other, out=self)

    def kron(self, other, op=NULL, out=None, **kwargs):
        """Kronecker product.

        """
        if out is None:
            out = self.__class__.from_type(
                self.type,
                self.nrows*other.nrows,
                self.ncols*other.ncols)
        if op is NULL:
            op = self.type.mult_op
        mask, semiring, accum, desc = self._get_args(**kwargs)

        _check(lib.GxB_kron(
            out.matrix[0],
            mask,
            accum,
            op,
            self.matrix[0],
            other.matrix[0],
            desc))
        return out

    def slice_matrix(self, rindex=None, cindex=None, **kwargs):
        """Slice a submatrix.

        """
        mask, semiring, accum, desc = self._get_args(**kwargs)
        result_nrows = self.ncols if desc in descriptor.T_A else self.nrows
        result_ncols = self.nrows if desc in descriptor.T_A else self.ncols

        I, ni, isize = _build_range(rindex, result_nrows - 1)
        J, nj, jsize = _build_range(cindex, result_ncols - 1)
        if isize is None:
            isize = result_nrows
        if jsize is None:
            jsize = result_ncols

        result = self.__class__.from_type(self.type, isize, jsize)
        _check(lib.GrB_Matrix_extract(
            result.matrix[0],
            mask,
            accum,
            self.matrix[0],
            I,
            ni,
            J,
            nj,
            desc))
        return result

    def slice_vector(self, index, vslice=None, **kwargs):
        """Slice a subvector.

        """
        new_vec = ffi.new('GrB_Vector*')
        _check(lib.GrB_Vector_new(
            new_vec,
            self.type.gb_type,
            self.ncols))

        mask, semiring, accum, desc = self._get_args(**kwargs)
        stop_val = self.nrows if desc in descriptor.T_A else self.ncols
        I, ni, size = _build_range(vslice, stop_val)

        _check(lib.GrB_Col_extract(
            new_vec[0],
            mask,
            accum,
            self.matrix[0],
            I,
            ni,
            index,
            desc
            ))
        return Vector(new_vec, self.type)

    def __getitem__(self, index):
        if isinstance(index, int):
            # a[3] extract single row
            return self.slice_vector(index, None, desc=descriptor.tooo)
        if isinstance(index, slice):
            # a[3:] extract submatrix of rows
            return self.slice_matrix(index, None)

        if isinstance(index, Matrix):
            return self.slice_matrix(mask=index)

        if not isinstance(index, (tuple, list)):
            raise TypeError

        i0 = index[0]
        i1 = index[1]
        if isinstance(i0, int) and isinstance(i1, int):
            # a[3,3] extract single element
            result = self.type.ffi.new(self.type.ptr)
            _check(self.type.Matrix_extractElement(
                result,
                self.matrix[0],
                index[0],
                index[1]))
            return self.type.to_value(result[0])

        if isinstance(i0, int) and isinstance(i1, slice):
            # a[3,:] extract slice of row vector
            return self.slice_vector(i0, i1, desc=descriptor.tooo)

        if isinstance(i0, slice) and isinstance(i1, int):
            # a[:,3] extract slice of col vector
            return self.slice_vector(i1, i0)

        if isinstance(i0, slice) and isinstance(i1, slice):
            # a[:,:] extract submatrix
            return self.slice_matrix(i0, i1)

    def assign_col(self, index, value, vslice=None, desc=descriptor.oooo):
        """Assign a vector to a column.

        """
        stop_val = self.ncols if desc in descriptor.T_A else self.nrows
        I, ni, size = _build_range(vslice, stop_val)

        _check(lib.GrB_Col_assign(
            self.matrix[0],
            NULL,
            NULL,
            value.vector[0],
            I,
            ni,
            index,
            desc
            ))

    def assign_row(self, index, value, vslice=None, desc=descriptor.oooo):
        """Assign a vector to a row.

        """
        stop_val = self.nrows if desc in descriptor.T_A else self.ncols
        I, ni, size = _build_range(vslice, stop_val)

        _check(lib.GrB_Row_assign(
            self.matrix[0],
            NULL,
            NULL,
            value.vector[0],
            index,
            I,
            ni,
            desc
            ))

    def assign_matrix(self, value, rindex=None, cindex=None, desc=descriptor.oooo):
        """Assign a submatrix.

        """
        I, ni, isize = _build_range(rindex, self.nrows - 1)
        J, nj, jsize = _build_range(cindex, self.ncols - 1)
        if isize is None:
            isize = self.nrows
        if jsize is None:
            jsize = self.ncols

        _check(lib.GrB_Matrix_assign(
            self.matrix[0],
            NULL,
            NULL,
            value.matrix[0],
            I,
            ni,
            J,
            nj,
            NULL))

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
            if isinstance(value, (bool, int, float)):
                # scalar assignment TODO
                raise NotImplementedError

        if isinstance(index, Matrix):
            if isinstance(value, Matrix):
                # A[M] = B masked matrix assignment
                raise NotImplementedError
            if not isinstance(value, (bool, int, float)):
                raise TypeError
            # A[M] = s masked scalar assignment
            scalar_type = types._gb_from_type(type(value))
            _check(scalar_type.Matrix_assignScalar(
                self.matrix[0],
                index.matrix[0],
                NULL,
                value,
                lib.GrB_ALL,
                0,
                lib.GrB_ALL,
                0,
                NULL
                ))
            return

        if not isinstance(index, (tuple, list)):
            raise TypeError

        i0 = index[0]
        i1 = index[1]
        if isinstance(i0, int) and isinstance(i1, int):
            val = self.type.from_value(value)
            _check(self.type.Matrix_setElement(
                self.matrix[0],
                val,
                i0,
                i1))
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
            if isinstance(value, (bool, int, float)):
                I, ni, isize = _build_range(i0, self.nrows - 1)
                J, nj, jsize = _build_range(i1, self.ncols - 1)
                scalar_type = types._gb_from_type(type(value))
                _check(scalar_type.Matrix_assignScalar(
                    self.matrix[0],
                    NULL,
                    NULL,
                    value,
                    I,
                    ni,
                    J,
                    nj,
                    NULL
                    ))
                return


            # a[:,:] assign submatrix
            self.assign_matrix(value, i0, i1)
            return
        raise TypeError('Unknown index or value for matrix assignment.')

    def __repr__(self):
        return '<Matrix (%sx%s : %s:%s)>' % (
            self.nrows,
            self.ncols,
            self.nvals,
            self.type.__name__)
