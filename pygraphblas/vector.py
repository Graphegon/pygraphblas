
from .base import (
    lib,
    ffi,
    _check,
    _gb_from_type,
    _default_add_op,
    _default_mul_op,
    )

class Vector:

    def __init__(self, vec):
        self.vector = vec

    def __del__(self):
        _check(lib.GrB_Vector_free(self.vector))

    def __eq__(self, other):
        result = ffi.new('_Bool*')
        _check(lib.LAGraph_Vector_isequal(
            result,
            self.vector[0],
            other.vector[0],
            ffi.NULL))
        return result[0]

    @classmethod
    def from_type(cls, py_type, size=0):
        new_vec = ffi.new('GrB_Vector*')
        gb_type = _gb_from_type(py_type)
        _check(lib.GrB_Vector_new(new_vec, gb_type, size))
        return cls(new_vec)

    @classmethod
    def from_lists(cls, I, V, size=None):
        if not size:
            size = len(I)
        # TODO use ffi and GrB_Vector_build
        m = cls.from_type(int, size)
        for i, v in zip(I, V):
            m[i] = v
        return m

    @classmethod
    def from_list(cls, I):
        size = len(I)
        # TODO use ffi and GrB_Vector_build
        m = cls.from_type(int, size)
        for i, v in enumerate(I):
            m[i] = v
        return m

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

    @property
    def gb_type(self):
        typ = ffi.new('GrB_Type*')
        _check(lib.GxB_Vector_type(
            typ,
            self.vector[0]))
        return typ[0]

    def ewise_add(self, other, out=None,
                  mask=None, accum=None, add_op=None, desc=None):
        if mask is None:
            mask = ffi.NULL
        if accum is None:
            accum = ffi.NULL
        if add_op is None:
            add_op = lib.GrB_PLUS_INT64
        if desc is None:
            desc = ffi.NULL
        if out is None:
            _out = ffi.new('GrB_Vector*')
            _check(lib.GrB_Vector_new(_out, self.gb_type, self.size))
            out = Vector(_out)
        _check(lib.GrB_eWiseAdd_Vector_BinaryOp(
            out.vector[0],
            mask,
            accum,
            add_op,
            self.vector[0],
            other.vector[0],
            desc))
        return out

    def ewise_mult(self, other, out=None,
                  mask=None, accum=None, mult_op=None, desc=None):
        if mask is None:
            mask = ffi.NULL
        if accum is None:
            accum = ffi.NULL
        if mult_op is None:
            mult_op = lib.GrB_TIMES_INT64
        if desc is None:
            desc = ffi.NULL
        if out is None:
            _out = ffi.new('GrB_Vector*')
            _check(lib.GrB_Vector_new(_out, self.gb_type, self.size))
            out = Vector(_out)
        _check(lib.GrB_eWiseMult_Vector_BinaryOp(
            out.vector[0],
            mask,
            accum,
            mult_op,
            self.vector[0],
            other.vector[0],
            desc))
        return out

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
