
from .semiring import Semiring
from .base import (
    lib,
    ffi,
    _check,
    _gb_from_type,
    _default_add_op,
    _default_mul_op,
    _build_range,
)
from . import descriptor

NULL = ffi.NULL

class Vector:

    _type_funcs = {
        lib.GrB_BOOL: {
            'C': '_Bool',
            'setElement': lib.GrB_Vector_setElement_BOOL,
            'extractElement': lib.GrB_Vector_extractElement_BOOL,
            'extractTuples': lib.GrB_Vector_extractTuples_BOOL,
            'add_op': lib.GrB_PLUS_BOOL,
            'mult_op': lib.GrB_TIMES_BOOL,
            'semiring': lib.GxB_LOR_LAND_BOOL,
            'assignScalar': lib.GrB_Vector_assign_BOOL,
        },
        lib.GrB_INT64: {
            'C': 'int64_t',
            'setElement': lib.GrB_Vector_setElement_INT64,
            'extractElement': lib.GrB_Vector_extractElement_INT64,
            'extractTuples': lib.GrB_Vector_extractTuples_INT64,
            'add_op': lib.GrB_PLUS_INT64,
            'mult_op': lib.GrB_TIMES_INT64,
            'semiring': lib.GxB_PLUS_TIMES_INT64,
            'assignScalar': lib.GrB_Vector_assign_INT64,
        },
        lib.GrB_FP64: {
            'C': 'double',
            'setElement': lib.GrB_Vector_setElement_FP64,
            'extractElement': lib.GrB_Vector_extractElement_FP64,
            'extractTuples': lib.GrB_Vector_extractTuples_FP64,
            'add_op': lib.GrB_PLUS_FP64,
            'mult_op': lib.GrB_TIMES_FP64,
            'semiring': lib.GxB_PLUS_TIMES_FP64,
            'assignScalar': lib.GrB_Vector_assign_FP64,
        },
    }

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
            NULL))
        return result[0]

    @classmethod
    def from_type(cls, py_type, size=0):
        new_vec = ffi.new('GrB_Vector*')
        gb_type = _gb_from_type(py_type)
        _check(lib.GrB_Vector_new(new_vec, gb_type, size))
        return cls(new_vec)

    @classmethod
    def from_lists(cls, I, V, size=None):
        assert len(I) == len(V)
        assert len(I) > 0 # must be non empty
        if not size:
            size = len(I)
        # TODO use ffi and GrB_Vector_build
        m = cls.from_type(type(V[0]), size)
        for i, v in zip(I, V):
            m[i] = v
        return m

    @classmethod
    def from_list(cls, I):
        size = len(I)
        assert size > 0
        # TODO use ffi and GrB_Vector_build
        m = cls.from_type(type(I[0]), size)
        for i, v in enumerate(I):
            m[i] = v
        return m

    @classmethod
    def dup(cls, vec):
        new_vec = ffi.new('GrB_Vector*')
        _check(lib.GrB_Vector_dup(new_vec, vec.vector[0]))
        return cls(new_vec)

    def to_lists(self):
        tf = self._type_funcs[self.gb_type]
        C = tf['C']
        I = ffi.new('GrB_Index[]', self.nvals)
        V = ffi.new(C + '[]', self.nvals)
        n = ffi.new('GrB_Index*')
        n[0] = self.nvals
        func = tf['extractTuples']
        _check(func(
            I,
            V,
            n,
            self.vector[0]
            ))
        return [list(I), list(V)]

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
                  mask=NULL, accum=NULL, add_op=NULL, desc=descriptor.oooo):
        if add_op is NULL:
            add_op = self._type_funcs[self.gb_type]['add_op']
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

    def vxm(self, other, out=None,
            mask=NULL, accum=NULL, semiring=NULL, desc=descriptor.oooo):
        from .matrix import Matrix
        if out is None:
            out = Vector.from_type(self.gb_type, self.size)
        elif not isinstance(out, Vector):
            raise TypeError('Output argument must be Vector.')
        if isinstance(mask, Matrix):
            mask = mask.matrix[0]
        if semiring is NULL:
            semiring = self._type_funcs[self.gb_type]['semiring']
        elif isinstance(semiring, Semiring):
            semiring = semiring.semiring
        _check(lib.GrB_vxm(
            out.vector[0],
            mask,
            accum,
            semiring,
            self.vector[0],
            other.matrix[0],
            desc))
        return out

    def __matmul__(self, other):
        return self.vxm(other)

    def __imatmul__(self, other):
        return self.vxm(other, out=self)

    def __add__(self, other):
        return self.ewise_add(other)

    def __iadd__(self, other):
        return self.ewise_add(other, out=self)

    def __mul__(self, other):
        return self.ewise_mult(other)

    def __imul__(self, other):
        return self.ewise_mult(other, out=self)

    def ewise_mult(self, other, out=None,
                   mask=NULL, accum=NULL, mult_op=NULL, desc=descriptor.oooo):
        if mult_op is NULL:
            mult_op = self._type_funcs[self.gb_type]['mult_op']
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

    def reduce_bool(self, accum=NULL, monoid=NULL, desc=descriptor.oooo):
        if monoid is NULL:
            monoid = lib.GxB_LOR_BOOL_MONOID
        result = ffi.new('_Bool*')
        _check(lib.GrB_Vector_reduce_BOOL(
            result,
            accum,
            monoid,
            self.vector[0],
            desc))
        return result[0]

    def reduce_int(self, accum=NULL, monoid=NULL, desc=descriptor.oooo):
        if monoid is NULL:
            monoid = lib.GxB_PLUS_INT64_MONOID
        result = ffi.new('int64_t*')
        _check(lib.GrB_Vector_reduce_INT64(
            result,
            accum,
            monoid,
            self.vector[0],
            desc))
        return result[0]

    def reduce_float(self, accum=NULL, monoid=NULL, desc=descriptor.oooo):
        if monoid is NULL:
            monoid = lib.GxB_PLUS_FP64_MONOID
        result = ffi.new('double*')
        _check(lib.GrB_Vector_reduce_FP64(
            result,
            accum,
            monoid,
            self.vector[0],
            desc))
        return result[0]

    def apply(self, op, out=None, mask=NULL, accum=NULL, desc=descriptor.oooo):
        if out is None:
            out = Vector.from_type(self.gb_type, self.size)
        _check(lib.GrB_Vector_apply(
            out.vector[0],
            mask,
            accum,
            op,
            self.vector[0],
            desc
            ))
        return out

    def __setitem__(self, index, value):
        tf = self._type_funcs[self.gb_type]
        if isinstance(index, int):
            C = tf['C']
            func = tf['setElement']
            _check(func(
                self.vector[0],
                ffi.cast(C, value),
                index))
            return
        if isinstance(index, slice):
            if isinstance(value, Vector):
                I, ni, size = _build_range(index, self.size - 1)
                _check(lib.GrB_Vector_assign(
                    self.vector[0],
                    NULL,
                    NULL,
                    value.vector[0],
                    I,
                    ni,
                    NULL
                    ))
                return
            if isinstance(value, (bool, int, float)):
                scalar_type = _gb_from_type(type(value))
                func = self._type_funcs[scalar_type]['assignScalar']
                I, ni, size = _build_range(index, self.size - 1)
                _check(func(
                    self.vector[0],
                    NULL,
                    NULL,
                    value,
                    I,
                    ni,
                    NULL
                    ))
                return
        raise TypeError('Unknown index or value for vector assignment.')

    def __getitem__(self, index):
        if isinstance(index, int):
            tf = self._type_funcs[self.gb_type]
            C = tf['C']
            func = tf['extractElement']
            result = ffi.new(C + '*')
            _check(func(
                result,
                self.vector[0],
                ffi.cast('GrB_Index', index)))
            return result[0]
        if isinstance(index, slice):
            I, ni, size = _build_range(index, self.size - 1)
            if size is None:
                size = self.size
            result = Vector.from_type(self.gb_type, size)
            _check(lib.GrB_Vector_extract(
                result.vector[0],
                NULL,
                NULL,
                self.vector[0],
                I,
                ni,
                NULL))
            return result

    def __repr__(self):
        return '<Vector (%s: %s)>' % (self.size, self.nvals)
