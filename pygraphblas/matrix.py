from .base import lib, ffi, _check, _gb_from_type
from .vector import Vector

class Matrix:

    _type_funcs = {
        lib.GrB_BOOL: {
            'C': '_Bool',
            'setElement': lib.GrB_Matrix_setElement_BOOL,
            'extractElement': lib.GrB_Matrix_extractElement_BOOL,
            'add_op': lib.GrB_PLUS_BOOL,
            'mult_op': lib.GrB_TIMES_BOOL,
            'monoid': lib.GxB_LOR_BOOL_MONOID,
            'semiring': lib.GxB_LOR_LAND_BOOL,
        },
        lib.GrB_INT64: {
            'C': 'int64_t',
            'setElement': lib.GrB_Matrix_setElement_INT64,
            'extractElement': lib.GrB_Matrix_extractElement_INT64,
            'add_op': lib.GrB_PLUS_INT64,
            'mult_op': lib.GrB_TIMES_INT64,
            'monoid': lib.GxB_PLUS_INT64_MONOID,
            'semiring': lib.GxB_PLUS_TIMES_INT64,
        },
        lib.GrB_FP64: {
            'C': 'double',
            'setElement': lib.GrB_Matrix_setElement_FP64,
            'extractElement': lib.GrB_Matrix_extractElement_FP64,
            'add_op': lib.GrB_PLUS_FP64,
            'mult_op': lib.GrB_TIMES_FP64,
            'monoid': lib.GxB_PLUS_FP64_MONOID,
            'semiring': lib.GxB_PLUS_TIMES_FP64,
        },
    }
    def __init__(self, matrix):
        self.matrix = matrix

    def __del__(self):
        _check(lib.GrB_Matrix_free(self.matrix))

    def __eq__(self, other):
        result = ffi.new('_Bool*')
        _check(lib.LAGraph_isequal(
            result,
            self.matrix[0],
            other.matrix[0],
            ffi.NULL))
        return result[0]

    @classmethod
    def from_type(cls, py_type, nrows=0, ncols=0):
        new_mat = ffi.new('GrB_Matrix*')
        gb_type = _gb_from_type(py_type)
        _check(lib.GrB_Matrix_new(new_mat, gb_type, nrows, ncols))
        return cls(new_mat)

    @classmethod
    def dup(cls, mat):
        new_mat = ffi.new('GrB_Matrix*')
        _check(lib.GrB_Matrix_dup(new_mat, mat.matrix[0]))
        return cls(new_mat)

    @classmethod
    def from_lists(cls, I, J, V, nrows=None, ncols=None):
        if not nrows:
            nrows = len(I)
        if not ncols:
            ncols = len(J)
        # TODO use ffi and GrB_Matrix_build
        m = cls.from_type(int, nrows, ncols)
        for i, j, v in zip(I, J, V):
            m[i, j] = v
        return m

    @classmethod
    def from_mm(cls, mm_file):
        m = ffi.new('GrB_Matrix*')
        _check(lib.LAGraph_mmread(m, mm_file))
        return cls(m)

    @property
    def gb_type(self):
        new_type = ffi.new('GrB_Type*')
        _check(lib.GxB_Matrix_type(new_type, self.matrix[0]))
        return new_type[0]

    @property
    def nrows(self):
        n = ffi.new('GrB_Index*')
        _check(lib.GrB_Matrix_ncols(n, self.matrix[0]))
        return n[0]

    @property
    def ncols(self):
        n = ffi.new('GrB_Index*')
        _check(lib.GrB_Matrix_nrows(n, self.matrix[0]))
        return n[0]

    @property
    def nvals(self):
        n = ffi.new('GrB_Index*')
        _check(lib.GrB_Matrix_nvals(n, self.matrix[0]))
        return n[0]

    def pattern(self):
        r = ffi.new('GrB_Matrix*')
        _check(lib.LAGraph_pattern(r, self.matrix[0]))
        return Matrix(r)

    def to_mm(self, fileobj):
        _check(lib.LAGraph_mmwrite(self.matrix[0], fileobj))

    def clear(self):
        _check(lib.GrB_Matrix_clear(self.matrix[0]))

    def resize(self, nrows, ncols):
        _check(lib.GxB_Matrix_resize(
            self.matrix[0],
            nrows,
            ncols))

    def ewise_add(self, other, out=None,
                  mask=None, accum=None, add_op=None, desc=None):
        if mask is None:
            mask = ffi.NULL
        if accum is None:
            accum = ffi.NULL
        if add_op is None:
            add_op = self._type_funcs[self.gb_type]['add_op']
        if desc is None:
            desc = ffi.NULL
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
                  mask=None, accum=None, mult_op=None, desc=None):
        if mask is None:
            mask = ffi.NULL
        if accum is None:
            accum = ffi.NULL
        if mult_op is None:
            mult_op = self._type_funcs[self.gb_type]['mult_op']
        if desc is None:
            desc = ffi.NULL
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

    def __add__(self, other):
        return self.ewise_add(other)

    def __iadd__(self, other):
        return self.ewise_add(other, out=self)

    def __mul__(self, other):
        return self.ewise_mult(other)

    def __imul__(self, other):
        return self.ewise_mult(other, out=self)

    def reduce_bool(self, accum=None, monoid=None, desc=None):
        if accum is None:
            accum = ffi.NULL
        if monoid is None:
            monoid = lib.GxB_LOR_BOOL_MONOID
        if desc is None:
            desc = ffi.NULL
        result = ffi.new('_Bool*')
        _check(lib.GrB_Matrix_reduce_BOOL(
            result,
            accum,
            monoid,
            self.matrix[0],
            desc))
        return result[0]

    def reduce_int(self, accum=None, monoid=None, desc=None):
        if accum is None:
            accum = ffi.NULL
        if monoid is None:
            monoid = lib.GxB_PLUS_INT64_MONOID
        if desc is None:
            desc = ffi.NULL
        result = ffi.new('int64_t*')
        _check(lib.GrB_Matrix_reduce_INT64(
            result,
            accum,
            monoid,
            self.matrix[0],
            desc))
        return result[0]

    def reduce_float(self, accum=None, monoid=None, desc=None):
        if accum is None:
            accum = ffi.NULL
        if monoid is None:
            monoid = lib.GxB_PLUS_FP64_MONOID
        if desc is None:
            desc = ffi.NULL
        result = ffi.new('double*')
        _check(lib.GrB_Matrix_reduce_FP64(
            result,
            accum,
            monoid,
            self.matrix[0],
            desc))
        return result[0]

    def reduce_vector(self, out=None, mask=None, accum=None, monoid=None, desc=None):
        if mask is None:
            mask = ffi.NULL
        if accum is None:
            accum = ffi.NULL
        if monoid is None:
            monoid = self._type_funcs[self.gb_type]['monoid']
        if desc is None:
            desc = ffi.NULL
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

    def mxm(self, other, out=None,
            mask=None, accum=None, semiring=None, desc=None):
        if out is None:
            out = Matrix.from_type(self.gb_type, self.nrows, other.ncols)
        if mask is None:
            mask = ffi.NULL
        if accum is None:
            accum = ffi.NULL
        if semiring is None:
            semiring = self._type_funcs[self.gb_type]['semiring']
        if desc is None:
            desc = ffi.NULL
        _check(lib.GrB_mxm(
            out.matrix[0],
            mask,
            accum,
            semiring,
            self.matrix[0],
            other.matrix[0],
            desc))
        return out

    def __matmul__(self, other):
        return self.mxm(other)

    def __imatmul__(self, other):
        return self.mxm(other, out=self)

    def slice_matrix(self, rindex=slice(None), cindex=slice(None), trans=False):
        desc = ffi.new('GrB_Descriptor*')
        if row:
            # transpose input to get row
            _check(lib.GrB_Descriptor_new(desc))
            _check(lib.GrB_Descriptor_set(
                desc[0],
                lib.GrB_INP0,
                lib.GrB_TRAN))
        else:
            desc[0] = ffi.NULL

        if isinstance(rindex, slice):
            I, ni = self._build_range(rindex, self.nrows)
        elif isinstance(rindex, int):
            pass
        if isinstance(cindex, slice):
            J, nj = self._build_range(cindex, self.ncols)
        elif isinstance(cindex, int):
            pass

    def slice_vector(self, index, vslice=None, transpose=False):
        """Slice a column or row vector out of the matrix.

        `index` is the column or row index to slice out.

        `vslice` is an optional slice object that applies to the
        vector.

        `transpose` if True, transpose the input to slice rows.
        Otherwise slice columns.

        """
        desc = ffi.new('GrB_Descriptor*')
        if transpose:
            # transpose input to get row
            _check(lib.GrB_Descriptor_new(desc))
            _check(lib.GrB_Descriptor_set(
                desc[0],
                lib.GrB_INP0,
                lib.GrB_TRAN))
        else:
            desc[0] = ffi.NULL

        new_vec = ffi.new('GrB_Vector*')
        _check(lib.GrB_Vector_new(
            new_vec,
            self.gb_type,
            self.ncols))

        stop_val = self.nrows if transpose else self.ncols
        I, ni = self._build_range(vslice, stop_val)

        _check(lib.GrB_Col_extract(
            new_vec[0],
            ffi.NULL,
            ffi.NULL,
            self.matrix[0],
            I,
            ni,
            index,
            desc[0]
            ))
        return Vector(new_vec)

    def __getitem__(self, index):
        if isinstance(index, int):
            # a[3] extract row vector
            return self.slice_vector(index, None, True)
        if isinstance(index, slice):
            # a[:] submatrix of rows
            return self.slice_matrix(index)

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

        if isinstance(i1, int) and isinstance(i1, (slice, tuple)):
            # a[3,:] extract slice of row vector
            return
        if isinstance(i1, (slice, tuple)) and isinstance(i1, int):
            # a[:,3] extract slice of col vector
            return
        if isinstance(i0, (slice, tuple)) and isinstance(i1, (slice, tuple)):
            # a[:,:] extract submatrix
            return

    def __setitem__(self, index, value):
        if isinstance(index, (int, slice)):
            return # TODO set row vector
        if isinstance(index, tuple):
            if isinstance(index[0], int) and isinstance(index[1], int):
                tf = self._type_funcs[self.gb_type]
                C = tf['C']
                func = tf['setElement']
                _check(func(
                    self.matrix[0],
                    ffi.cast(C, value),
                    index[0],
                    index[1]))

    def __repr__(self):
        return '<Matrix (%sx%s: %s)>' % (
            self.nrows,
            self.ncols,
            self.nvals)
