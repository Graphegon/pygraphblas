import sys
from random import randint
from .base import (
    lib,
    ffi,
    _check,
    _gb_from_type,
    _build_range,
)
from .vector import Vector
from .semiring import Semiring
from .unaryop import UnaryOp
from . import descriptor

NULL = ffi.NULL

class Matrix:
    """GraphBLAS Sparse Matrix

    This is a high-level wrapper around the low-level GrB_Matrix type.

    """

    _type_funcs = {
        lib.GrB_BOOL: {
            'C': '_Bool',
            'setElement': lib.GrB_Matrix_setElement_BOOL,
            'extractElement': lib.GrB_Matrix_extractElement_BOOL,
            'extractTuples': lib.GrB_Matrix_extractTuples_BOOL,
            'add_op': lib.GrB_PLUS_BOOL,
            'mult_op': lib.GrB_TIMES_BOOL,
            'monoid': lib.GxB_LOR_BOOL_MONOID,
            'semiring': lib.GxB_LOR_LAND_BOOL,
            'assignScalar': lib.GrB_Matrix_assign_BOOL,
        },
        lib.GrB_INT64: {
            'C': 'int64_t',
            'setElement': lib.GrB_Matrix_setElement_INT64,
            'extractElement': lib.GrB_Matrix_extractElement_INT64,
            'extractTuples': lib.GrB_Matrix_extractTuples_INT64,
            'add_op': lib.GrB_PLUS_INT64,
            'mult_op': lib.GrB_TIMES_INT64,
            'monoid': lib.GxB_PLUS_INT64_MONOID,
            'semiring': lib.GxB_PLUS_TIMES_INT64,
            'assignScalar': lib.GrB_Matrix_assign_INT64,
        },
        lib.GrB_FP64: {
            'C': 'double',
            'setElement': lib.GrB_Matrix_setElement_FP64,
            'extractElement': lib.GrB_Matrix_extractElement_FP64,
            'extractTuples': lib.GrB_Matrix_extractTuples_FP64,
            'add_op': lib.GrB_PLUS_FP64,
            'mult_op': lib.GrB_TIMES_FP64,
            'monoid': lib.GxB_PLUS_FP64_MONOID,
            'semiring': lib.GxB_PLUS_TIMES_FP64,
            'assignScalar': lib.GrB_Matrix_assign_FP64,
        },
    }
    def __init__(self, matrix):
        self.matrix = matrix

    def __del__(self):
        _check(lib.GrB_Matrix_free(self.matrix))

    @classmethod
    def from_type(cls, py_type, nrows=0, ncols=0):
        """Create an empty Matrix from the given type, number of rows, and
        number of columns.

        """
        new_mat = ffi.new('GrB_Matrix*')
        gb_type = _gb_from_type(py_type)
        _check(lib.GrB_Matrix_new(new_mat, gb_type, nrows, ncols))
        return cls(new_mat)

    @classmethod
    def dup(cls, mat):
        """Create an duplicate Matrix from the given argument.

        """
        new_mat = ffi.new('GrB_Matrix*')
        _check(lib.GrB_Matrix_dup(new_mat, mat.matrix[0]))
        return cls(new_mat)

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
        typ = type(V[0])
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
    def from_random(cls, gb_type, nrows, ncols, nvals,
                    make_pattern=False, make_symmetric=False,
                    make_skew_symmetric=False, make_hermitian=False,
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
            _gb_from_type(gb_type),
            nrows,
            ncols,
            nvals,
            make_pattern,
            make_symmetric,
            make_skew_symmetric,
            make_hermitian,
            no_diagonal,
            fseed))
        return cls(result)

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
        _check(lib.GrB_Matrix_ncols(n, self.matrix[0]))
        return n[0]

    @property
    def ncols(self):
        """Return the number of Matrix columns.

        """
        n = ffi.new('GrB_Index*')
        _check(lib.GrB_Matrix_nrows(n, self.matrix[0]))
        return n[0]

    @property
    def nvals(self):
        """Return the number of Matrix values.

        """
        n = ffi.new('GrB_Index*')
        _check(lib.GrB_Matrix_nvals(n, self.matrix[0]))
        return n[0]

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
        return Matrix(r)

    def to_mm(self, fileobj):
        """Write this matrix to a file using the Matrix Market format.

        """
        _check(lib.LAGraph_mmwrite(self.matrix[0], fileobj))

    def to_lists(self):
        """Extract the rows, columns and values of the Matrix as 3 lists.

        """
        tf = self._type_funcs[self.gb_type]
        C = tf['C']
        I = ffi.new('GrB_Index[]', self.nvals)
        J = ffi.new('GrB_Index[]', self.nvals)
        V = ffi.new(C + '[]', self.nvals)
        n = ffi.new('GrB_Index*')
        n[0] = self.nvals
        func = tf['extractTuples']
        _check(func(
            I,
            J,
            V,
            n,
            self.matrix[0]
            ))
        return [list(I), list(J), list(V)]

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

    def transpose(self, out=None, mask=NULL, accum=NULL, desc=descriptor.oooo):
        """ Transpose matrix. """
        if out is None:
            _out = ffi.new('GrB_Matrix*')
            _check(lib.GrB_Matrix_new(
                _out, self.gb_type, self.nrows, self.ncols))
            out = Matrix(_out)
        _check(lib.GrB_transpose(
            out.matrix[0],
            mask,
            accum,
            self.matrix[0],
            desc
            ))
        return out

    def ewise_add(self, other, out=None,
                  mask=NULL, accum=NULL, add_op=NULL, desc=descriptor.oooo):
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
            add_op = self._type_funcs[self.gb_type]['add_op']
        if out is None:
            _out = ffi.new('GrB_Matrix*')
            _check(lib.GrB_Matrix_new(
                _out, self.gb_type, self.nrows, self.ncols))
            out = Matrix(_out)
        _check(lib.GrB_eWiseAdd_Matrix_BinaryOp(
            out.matrix[0],
            mask,
            accum,
            add_op,
            self.matrix[0],
            other.matrix[0],
            desc))
        return out

    def ewise_mult(self, other, out=None,
                   mask=NULL, accum=NULL, mult_op=NULL, desc=descriptor.oooo):
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
            mult_op = self._type_funcs[self.gb_type]['mult_op']
        if out is None:
            _out = ffi.new('GrB_Matrix*')
            _check(lib.GrB_Matrix_new(
                _out, self.gb_type, self.nrows, self.ncols))
            out = Matrix(_out)
        _check(lib.GrB_eWiseMult_Matrix_BinaryOp(
            out.matrix[0],
            mask,
            accum,
            mult_op,
            self.matrix[0],
            other.matrix[0],
            desc))
        return out

    def __eq__(self, other):
        """Compare two matrices for equality.

        """
        result = ffi.new('_Bool*')
        _check(lib.LAGraph_isequal(
            result,
            self.matrix[0],
            other.matrix[0],
            NULL))
        return result[0]

    def __add__(self, other):
        return self.ewise_add(other)

    def __iadd__(self, other):
        return self.ewise_add(other, out=self)

    def __mul__(self, other):
        return self.ewise_mult(other)

    def __imul__(self, other):
        return self.ewise_mult(other, out=self)

    def reduce_bool(self, accum=NULL, monoid=NULL, desc=descriptor.oooo):
        """Reduce matrix to a boolean.

        """
        if monoid is NULL:
            monoid = lib.GxB_LOR_BOOL_MONOID
        result = ffi.new('_Bool*')
        _check(lib.GrB_Matrix_reduce_BOOL(
            result,
            accum,
            monoid,
            self.matrix[0],
            desc))
        return result[0]

    def reduce_int(self, accum=NULL, monoid=NULL, desc=descriptor.oooo):
        """Reduce matrix to an integer.

        """
        if monoid is NULL:
            monoid = lib.GxB_PLUS_INT64_MONOID
        result = ffi.new('int64_t*')
        _check(lib.GrB_Matrix_reduce_INT64(
            result,
            accum,
            monoid,
            self.matrix[0],
            desc))
        return result[0]

    def reduce_float(self, accum=NULL, monoid=NULL, desc=descriptor.oooo):
        """Reduce matrix to an float.

        """
        if monoid is NULL:
            monoid = lib.GxB_PLUS_FP64_MONOID
        result = ffi.new('double*')
        _check(lib.GrB_Matrix_reduce_FP64(
            result,
            accum,
            monoid,
            self.matrix[0],
            desc))
        return result[0]

    def reduce_vector(self, out=None,
                      mask=NULL, accum=NULL, monoid=NULL, desc=descriptor.oooo):
        """Reduce matrix to a vector.

        """
        if monoid is NULL:
            monoid = self._type_funcs[self.gb_type]['monoid']
        if out is None:
            out = Vector.from_type(self.gb_type, self.nrows)
        _check(lib.GrB_Matrix_reduce_Monoid(
            out.vector[0],
            mask,
            accum,
            monoid,
            self.matrix[0],
            desc))
        return out

    def apply(self, op, out=None, mask=NULL, accum=NULL, desc=descriptor.oooo):
        """Apply Unary op to matrix elements.

        """
        if out is None:
            out = Matrix.from_type(self.gb_type, self.nrows, self.ncols)
        if isinstance(op, UnaryOp):
            op = op.unaryop
        _check(lib.GrB_Matrix_apply(
            out.matrix[0],
            mask,
            accum,
            op,
            self.matrix[0],
            desc
            ))
        return out

    def select(self, op, out=None, mask=NULL, accum=NULL, thunk=NULL, desc=descriptor.oooo):
        if out is None:
            out = Matrix.from_type(self.gb_type, self.nrows, self.ncols)
        if isinstance(op, UnaryOp):
            op = op.unaryop
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

    def tril(self, diag=0):
        return self.select(lib.GxB_TRIL, thunk=ffi.new('int64_t*', diag))

    def triu(self, diag=0):
        return self.select(lib.GxB_TRIU, thunk=ffi.new('int64_t*', diag))

    def diag(self, diag=0):
        return self.select(lib.GxB_DIAG, thunk=ffi.new('int64_t*', diag))

    def offdiag(self, diag=0):
        return self.select(lib.GxB_OFFDIAG, thunk=ffi.new('int64_t*', diag))

    def nonzero(self, diag=0):
        return self.select(lib.GxB_NONZERO)

    def mxm(self, other, out=None,
            mask=NULL, accum=NULL, semiring=NULL, desc=descriptor.oooo):
        """Matrix-matrix multiply.

        """
        if out is None:
            out = Matrix.from_type(self.gb_type, self.nrows, other.ncols)
        if isinstance(mask, Matrix):
            mask = mask.matrix[0]
        if semiring is NULL:
            semiring = self._type_funcs[self.gb_type]['semiring']
        elif isinstance(semiring, Semiring):
            semiring = semiring.semiring
        _check(lib.GrB_mxm(
            out.matrix[0],
            mask,
            accum,
            semiring,
            self.matrix[0],
            other.matrix[0],
            desc))
        return out

    def mxv(self, other, out=None,
            mask=NULL, accum=NULL, semiring=NULL, desc=descriptor.oooo):
        """Matrix-vector multiply.

        """
        if out is None:
            out = Vector.from_type(self.gb_type, self.ncols)
        elif not isinstance(out, Vector):
            raise TypeError('Output hand argument must be Vector.')
        if isinstance(mask, Matrix):
            mask = mask.matrix[0]
        if semiring is NULL:
            semiring = self._type_funcs[self.gb_type]['semiring']
        elif isinstance(semiring, Semiring):
            semiring = semiring.semiring
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

    def __imatmul__(self, other):
        return self.mxm(other, out=self)

    def kron(self, other, out=None,
             mask=NULL, accum=NULL, op=NULL, desc=descriptor.oooo):
        """Kronecker product.

        """
        if out is None:
            out = Matrix.from_type(self.gb_type,
                                   self.nrows*other.nrows,
                                   self.ncols*other.ncols)
        if op is NULL:
            op = self._type_funcs[self.gb_type]['mult_op']
        _check(lib.GxB_kron(
            out.matrix[0],
            mask,
            accum,
            op,
            self.matrix[0],
            other.matrix[0],
            desc))
        return out

    def slice_matrix(self, rindex=None, cindex=None, desc=descriptor.oooo):
        """Slice a submatrix.

        """
        I, ni, isize = _build_range(rindex, self.nrows - 1)
        J, nj, jsize = _build_range(cindex, self.ncols - 1)
        if isize is None:
            isize = self.nrows
        if jsize is None:
            jsize = self.ncols

        result = Matrix.from_type(self.gb_type, isize, jsize)
        _check(lib.GrB_Matrix_extract(
            result.matrix[0],
            NULL,
            NULL,
            self.matrix[0],
            I,
            ni,
            J,
            nj,
            NULL))
        return result

    def slice_vector(self, index, vslice=None, desc=descriptor.oooo):
        """Slice a subvector.

        """
        new_vec = ffi.new('GrB_Vector*')
        _check(lib.GrB_Vector_new(
            new_vec,
            self.gb_type,
            self.ncols))

        stop_val = self.nrows if desc in descriptor.T_A else self.ncols
        I, ni, size = _build_range(vslice, stop_val)

        _check(lib.GrB_Col_extract(
            new_vec[0],
            NULL,
            NULL,
            self.matrix[0],
            I,
            ni,
            index,
            desc
            ))
        return Vector(new_vec)

    def __getitem__(self, index):
        if isinstance(index, int):
            # a[3] extract single row
            return self.slice_vector(index, None, descriptor.tooo)
        if isinstance(index, slice):
            # a[3:] extract submatrix of rows
            return self.slice_matrix(index, None)

        if not isinstance(index, (tuple, list)):
            raise TypeError

        i0 = index[0]
        i1 = index[1]
        if isinstance(i0, int) and isinstance(i1, int):
            # a[3,3] extract single element
            tf = self._type_funcs[self.gb_type]
            C = tf['C']
            func = tf['extractElement']
            result = ffi.new(C + '*')
            _check(func(
                result,
                self.matrix[0],
                index[0],
                index[1]))
            return result[0]

        if isinstance(i0, int) and isinstance(i1, slice):
            # a[3,:] extract slice of row vector
            return self.slice_vector(i0, i1, descriptor.tooo)

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
            # a[3] = assign single row  vector
            if isinstance(value, Vector):
                return self.assign_row(index, value)

        if isinstance(index, slice):
            # a[3:] = assign submatrix to rows
            if isinstance(value, Matrix):
                self.assign_matrix(value, index, None)
            if isinstance(value, (bool, int, float)):
                # scalar assignment TODO
                return

        if not isinstance(index, (tuple, list)):
            raise TypeError

        i0 = index[0]
        i1 = index[1]
        if isinstance(i0, int) and isinstance(i1, int):
            tf = self._type_funcs[self.gb_type]
            C = tf['C']
            func = tf['setElement']
            _check(func(
                self.matrix[0],
                ffi.cast(C, value),
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
            # a[:,:] assign submatrix
            self.assign_matrix(value, i0, i1)
            return
        raise TypeError('Unknown index or value for matrix assignment.')

    def __repr__(self):
        return '<Matrix (%sx%s: %s)>' % (
            self.nrows,
            self.ncols,
            self.nvals)
