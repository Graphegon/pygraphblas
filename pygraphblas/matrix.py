from itertools import chain
from . import lib, ffi, _check, _gb_from_type


class Matrix:

    def __init__(self, matrix):
        self.matrix = matrix

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
                    index[0],
                    index[1]))
                return result[0]

    def __repr__(self):
        return '<Matrix (%sx%s: %s)>' % (self.nrows, self.ncols, self.nvals)
