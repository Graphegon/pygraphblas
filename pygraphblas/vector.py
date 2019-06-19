
from . import lib, ffi, _check, _as_type


class Vector:

    def __init__(self, typ, ncols, nrows, data=None):
        self._A = ffi.new('GrB_Vector*')
        _check(lib.GrB_Vector_new(self._A, typ, ncols, nrows))
        # TODO: Optimize in C
        if data is not None:
            for i in data:
                pass

    @property
    def size(self):
        n = ffi.new('GrB_Index*')
        _check(lib.GrB_Vector_size(n, self._A[0]))
        return n[0]

    @property
    def nvals(self):
        n = ffi.new('GrB_Index*')
        _check(lib.GrB_Vector_nvals(n, self._A[0]))
        return n[0]

    def __getitem__(self, index):
        if isintance(index, int):
            pass # return row vector
        if isintance(index, tuple):
            pass # fancy numpy slicing
