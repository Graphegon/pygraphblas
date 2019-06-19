from itertools import chain
from . import lib, ffi, _check, _gb_from_type, _cffi_type_from


class Matrix:

    def __init__(self, A, nrows=None, ncols=None):
        if isinstance(A, Matrix):
            _B = ffi.new('GrB_Matrix*')
            _check(lib.GrB_Matrix_dup(_B, A.matrix[0]))
            self.matrix = _B
            self.gb_type = A.gb_type
        else:
            _A = ffi.new('GrB_Matrix*')
            gb_type = _gb_from_type(A)
            _check(lib.GrB_Matrix_new(_A, gb_type, nrows, ncols))
            self.matrix = _A
            self.gb_type = gb_type

    @property
    def matrix(self):
        return self._A

    @matrix.setter
    def matrix(self, A):
        self._A = A

    @property
    def gb_type(self):
        return self._gb_type

    @gb_type.setter
    def gb_type(self, gb_type):
        self._gb_type = gb_type

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

    def clear(self):
        _check(lib.GrB_Matrix_clear(self.matrix[0]))

    def resize(self, nrows, ncols):
        _check(lib.GxB_Matrix_resize(
            self.matrix[0],
            nrows,
            ncols))

    def __setitem__(self, index, value):
        if isinstance(index, int):
            return # TODO set row vector
        if isinstance(index, tuple):
            if isinstance(index[0], int) and isinstance(index[1], int):
                _check(lib.GrB_Matrix_setElement_INT64(
                    self.matrix[0],
                    ffi.cast('int64_t', value),
                    index[0],
                    index[1]))

    def __getitem__(self, index):
        if isinstance(index, int):
            return # TODO return row vector
        if isinstance(index, tuple):
            if isinstance(index[0], int) and isinstance(index[1], int):
                result = ffi.new('int64_t*')
                _check(lib.GrB_Matrix_extractElement_INT64(
                    result,
                    self.matrix[0],
                    ffi.cast('GrB_Index', index[0]),
                    ffi.cast('GrB_Index', index[1])))
                return result[0]

    def __repr__(self):
        return '<Matrix (%sx%s: %s)>' % (self.nrows, self.ncols, self.nvals)
