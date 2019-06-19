
from . import lib, ffi, _check


class Matrix:

    def __init__(self, typ, ncols, nrows):
        self._A = ffi.new('GrB_Matrix*')
        _check(lib.GrB_Matrix_new(self._A, typ, ncols, nrows))

    @property
    def nrows(self):
        n = ffi.new('GrB_Index*')
        _check(lib.GrB_Matrix_ncols(n, self._A[0]))
        return n[0]

    @property
    def ncols(self):
        n = ffi.new('GrB_Index*')
        _check(lib.GrB_Matrix_nrows(n, self._A[0]))
        return n[0]

    @property
    def nvals(self):
        n = ffi.new('GrB_Index*')
        _check(lib.GrB_Matrix_nvals(n, self._A[0]))
        return n[0]
