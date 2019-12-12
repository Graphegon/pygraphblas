import operator

from .base import (
    lib,
    ffi,
    NULL,
    _check,
    _get_bin_op,
    _gb_from_type,
    _default_add_op,
    _default_mul_op,
    _build_range,
)
from .semiring import Semiring, current_semiring
from .binaryop import BinaryOp, current_accum, current_binop
from .unaryop import UnaryOp
from .type_funcs import build_vector_type_funcs
from . import descriptor

__all__ = ['Vector']

class Vector:
    """GraphBLAS Sparse Vector

    This is a high-level wrapper around the low-level GrB_Vector type.

    """

    __slots__ = ('vector', '_funcs')

    def __init__(self, vec):
        self.vector = vec
        self._funcs = build_vector_type_funcs(self.gb_type)

    def __del__(self):
        _check(lib.GrB_Vector_free(self.vector))

    def __len__(self):
        return self.nvals

    def __iter__(self):
        nvals = self.nvals
        _nvals = ffi.new('GrB_Index[1]', [nvals])
        I = ffi.new('GrB_Index[%s]' % nvals)
        X = ffi.new('%s[%s]' % (self._funcs.C, nvals))
        _check(self._funcs.extractTuples(
            I,
            X,
            _nvals,
            self.vector[0]
            ))
        return zip(I, X)

    def iseq(self, other):
        result = ffi.new('_Bool*')
        _check(lib.LAGraph_Vector_isequal(
            result,
            self.vector[0],
            other.vector[0],
            NULL))
        return result[0]

    def isne(self, other):
        return not self.iseq(other)

    @classmethod
    def from_type(cls, py_type, size=0):
        """Create an empty Vector from the given type and size.

        """
        new_vec = ffi.new('GrB_Vector*')
        gb_type = _gb_from_type(py_type)
        _check(lib.GrB_Vector_new(new_vec, gb_type, size))
        return cls(new_vec)

    @classmethod
    def from_lists(cls, I, V, size=None):
        """Create a new vector from the given lists of indices and values.  If
        size is not provided, it is computed from the max values of
        the provides size indices.

        """
        assert len(I) == len(V)
        assert len(I) > 0 # must be non empty
        if not size:
            size = max(I) + 1
        # TODO option to use ffi and GrB_Vector_build
        m = cls.from_type(type(V[0]), size)
        for i, v in zip(I, V):
            m[i] = v
        return m

    @classmethod
    def from_list(cls, I):
        """Create a new dense vector from the given lists of values.

        """
        size = len(I)
        assert size > 0
        # TODO use ffi and GrB_Vector_build
        m = cls.from_type(type(I[0]), size)
        for i, v in enumerate(I):
            m[i] = v
        return m

    @classmethod
    def dup(cls, vec):
        """Create an duplicate Vector from the given argument.

        """
        new_vec = ffi.new('GrB_Vector*')
        _check(lib.GrB_Vector_dup(new_vec, vec.vector[0]))
        return cls(new_vec)

    def to_lists(self):
        """Extract the indices and values of the Vector as 2 lists.

        """
        tf = self._funcs
        C = tf.C
        I = ffi.new('GrB_Index[]', self.nvals)
        V = ffi.new(C + '[]', self.nvals)
        n = ffi.new('GrB_Index*')
        n[0] = self.nvals
        func = tf.extractTuples
        _check(func(
            I,
            V,
            n,
            self.vector[0]
            ))
        return [list(I), list(V)]

    @property
    def size(self):
        """Return the size of the vector.

        """
        n = ffi.new('GrB_Index*')
        _check(lib.GrB_Vector_size(n, self.vector[0]))
        return n[0]

    @property
    def shape(self):
        """Numpy-like description of vector shape.

        """
        return (self.size,)

    @property
    def nvals(self):
        """Return the number of values in the vector.

        """
        n = ffi.new('GrB_Index*')
        _check(lib.GrB_Vector_nvals(n, self.vector[0]))
        return n[0]

    @property
    def gb_type(self):
        """Return the GraphBLAS low-level type object of the Vector.

        """
        typ = ffi.new('GrB_Type*')
        _check(lib.GxB_Vector_type(
            typ,
            self.vector[0]))
        return typ[0]

    def full(self, identity=None):
        B = self.__class__.from_type(self.gb_type, self.size)
        if identity is None:
            identity = self._funcs.identity

        _check(self._funcs.assignScalar(
            B.vector[0],
            NULL,
            NULL,
            identity,
            lib.GrB_ALL,
            0,
            NULL))
        return self.eadd(B, self._funcs.first)

    def compare(self, other, op, strop):
        C = self.__class__.from_type(bool, self.size)
        if isinstance(other, (bool, int, float)):
            if op(other, 0):
                B = self.__class__.dup(self)
                B[:] = other
                self.emult(B, strop, out=C)
                return C
            else:
                self.select(strop, other).apply(lib.GxB_ONE_BOOL, out=C)
                return C
        elif isinstance(other, Vector):
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

    def eadd(self, other, add_op=NULL, out=None,
                  mask=NULL, accum=NULL, desc=descriptor.oooo):
        """Element-wise addition with other vector.

        Element-wise addition applies a binary operator element-wise
        on two vectors A and B, for all entries that appear in the set
        intersection of the patterns of A and B.  Other operators
        other than addition can be used.

        The pattern of the result of the element-wise addition is
        the set union of the pattern of A and B. Entries in neither in
        A nor in B do not appear in the result.

        """
        if add_op is NULL:
            add_op = current_binop.get(self._funcs.add_op)
        elif isinstance(add_op, str):
            add_op = _get_bin_op(add_op, self._funcs)
        if accum is NULL:
            accum = current_accum.get(NULL)
        if isinstance(accum, BinaryOp):
            accum = accum.get_binaryop(self, other)
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

    def emult(self, other, mult_op=NULL, out=None,
                   mask=NULL, accum=NULL, desc=descriptor.oooo):
        """Element-wise multiplication with other vector.

        Element-wise multiplication applies a binary operator
        element-wise on two vectors A and B, for all entries that
        appear in the set intersection of the patterns of A and B.
        Other operators other than addition can be used.

        The pattern of the result of the element-wise multiplication
        is exactly this set intersection. Entries in A but not B, or
        visa versa, do not appear in the result.

        """
        if mult_op is NULL:
            mult_op = current_binop.get(self._funcs.mult_op)
        elif isinstance(mult_op, str):
            mult_op = _get_bin_op(mult_op, self._funcs)
        if accum is NULL:
            accum = current_accum.get(NULL)
        if isinstance(accum, BinaryOp):
            accum = accum.get_binaryop(self, other)
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

    def vxm(self, other, out=None,
            mask=NULL, accum=NULL, semiring=NULL, desc=descriptor.oooo):
        """Vector-Matrix multiply.
        """
        from .matrix import Matrix
        if out is None:
            out = Vector.from_type(self.gb_type, self.size)
        elif not isinstance(out, Vector):
            raise TypeError('Output argument must be Vector.')
        if isinstance(mask, Vector):
            mask = mask.vector[0]
        if semiring is NULL:
            semiring = current_semiring.get(self._funcs.semiring)
        if isinstance(semiring, Semiring):
            semiring = semiring.get_semiring(self)
        if accum is NULL:
            accum = current_accum.get(NULL)
        if isinstance(accum, BinaryOp):
            accum = accum.get_binaryop(self, other)
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
        return self.eadd(other)

    def __iadd__(self, other):
        return self.eadd(other, out=self)

    def __mul__(self, other):
        return self.emult(other)

    def __imul__(self, other):
        return self.emult(other, out=self)

    def __invert__(self):
        return self.apply(self._funcs.invert)

    def __abs__(self):
        return self.apply(self._funcs.abs_)

    def clear(self):
        _check(lib.GrB_Vector_clear(self.vector[0]))

    def resize(self, size):
        _check(lib.GxB_Vector_resize(
            self.vector[0],
            size))

    def _get_args(self, mask=NULL, accum=NULL, monoid=NULL, desc=descriptor.oooo):
        if monoid is NULL:
            monoid = self._funcs.monoid
        if accum is NULL:
            accum = current_accum.get(NULL)
        if isinstance(accum, BinaryOp):
            accum = accum.get_binaryop(self)
        if isinstance(mask, Vector):
            mask = mask.vector[0]
        return mask, monoid, accum, desc


    def reduce_bool(self, **kwargs):
        """Reduce vector to a boolean.

        """
        mask, monoid, accum, desc = self._get_args(**kwargs)
        result = ffi.new('_Bool*')
        _check(lib.GrB_Vector_reduce_BOOL(
            result,
            accum,
            monoid,
            self.vector[0],
            desc))
        return result[0]

    def reduce_int(self, **kwargs):
        """Reduce vector to a integer.

        """
        mask, monoid, accum, desc = self._get_args(**kwargs)
        result = ffi.new('int64_t*')
        _check(lib.GrB_Vector_reduce_INT64(
            result,
            accum,
            monoid,
            self.vector[0],
            desc))
        return result[0]

    def reduce_float(self, **kwargs):
        """Reduce vector to a float.

        """
        mask, monoid, accum, desc = self._get_args(**kwargs)
        result = ffi.new('double*')
        _check(lib.GrB_Vector_reduce_FP64(
            result,
            accum,
            monoid,
            self.vector[0],
            desc))
        return result[0]

    def apply(self, op, out=None, **kwargs):
        """Apply Unary op to vector elements.

        """
        if out is None:
            out = Vector.from_type(self.gb_type, self.size)
        if isinstance(op, UnaryOp):
            op = op.unaryop

        mask, monoid, accum, desc = self._get_args(**kwargs)
        _check(lib.GrB_Vector_apply(
            out.vector[0],
            mask,
            accum,
            op,
            self.vector[0],
            desc
            ))
        return out

    def select(self, op, out=None, thunk=NULL, **kwargs):
        if out is None:
            out = Vector.from_type(self.gb_type, self.size)
        if isinstance(op, UnaryOp):
            op = op.unaryop

        mask, monoid, accum, desc =self._get_args(**kwargs)
        _check(lib.GxB_Vector_select(
            out.vector[0],
            mask,
            accum,
            op,
            self.vector[0],
            thunk,
            desc
            ))
        return out

    def to_dense(self, _id=None):
        out = ffi.new('GrB_Vector*')
        if _id is None:
            C = self._funcs.C
            _id = ffi.new(C + '*', 0)
        _check(lib.LAGraph_Vector_to_dense(
            out,
            self.vector[0],
            _id))
        return Vector(out)

    def __setitem__(self, index, value):
        mask = NULL
        desc = NULL
        tf = self._funcs
        if isinstance(index, int):
            C = tf.C
            func = tf.setElement
            _check(func(
                self.vector[0],
                ffi.cast(C, value),
                index))
            return
        if isinstance(index, tuple):
            if len(index) == 2:
                index, mask = index
            elif len(index) == 3:
                index, mask, desc = index
        if isinstance(mask, Vector):
            mask = mask.vector[0]
            
        if isinstance(index, slice):
            if isinstance(value, Vector):
                I, ni, size = _build_range(index, self.size - 1)
                _check(lib.GrB_Vector_assign(
                    self.vector[0],
                    mask,
                    NULL,
                    value.vector[0],
                    I,
                    ni,
                    desc
                    ))
                return
            if isinstance(value, (bool, int, float)):
                scalar_type = _gb_from_type(type(value))
                tf = build_vector_type_funcs(scalar_type)
                I, ni, size = _build_range(index, self.size - 1)
                _check(tf.assignScalar(
                    self.vector[0],
                    mask,
                    NULL,
                    value,
                    I,
                    ni,
                    desc
                    ))
                return
        raise TypeError('Unknown index or value for vector assignment.')

    def __getitem__(self, index):
        if isinstance(index, int):
            tf = self._funcs
            C = tf.C
            func = tf.extractElement
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
