
from .base import lib, ffi, _check, _gb_from_type

class Vector:

    def __init__(self, vec):
        self.vector = vec

    def __del__(self):
        _check(lib.GrB_Vector_free(self.vector[0]))

    @classmethod
    def from_type(cls, py_type, size=0):
        new_vec = ffi.new('GrB_Vector*')
        gb_type = _gb_from_type(py_type)
        _check(lib.GrB_Vector_new(new_vec, gb_type, size))
        return cls(new_vec)

    @classmethod
    def dup(cls, vec):
        new_vec = ffi.new('GrB_Vector*')
        _check(lib.GrB_Vector_dup(new_vec, vec.vector[0]))
        return cls(new_vec)

    @property
    def size(self):
        n = ffi.new('GrB_Index*')
        _check(lib.GrB_Vector_size(n, self.vector[0]))
        return n[0]

    @property
    def nvals(self):
        n = ffi.new('GrB_Index*')
        _check(lib.GrB_Vector_nvals(n, self.vector[0]))
        return n[0]

    def clear(self):
        _check(lib.GrB_Vector_clear(self.vector[0]))

    def resize(self, size):
        _check(lib.GxB_Vector_resize(
            self.vector[0],
            size))

    def __setitem__(self, index, value):
        _check(lib.GrB_Vector_setElement_INT64(
            self.vector[0],
            ffi.cast('int64_t', value),
            index))

    def __getitem__(self, index):
        result = ffi.new('int64_t*')
        _check(lib.GrB_Vector_extractElement_INT64(
            result,
            self.vector[0],
            ffi.cast('GrB_Index', index)))
        return result[0]

    def __repr__(self):
        return '<Vector (%s: %s)>' % (self.size, self.nvals)
