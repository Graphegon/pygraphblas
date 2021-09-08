"""High level wrapper around GraphBLAS Vectors.

"""
import random
import sys
import operator
import weakref
from array import array
from functools import partial
import numpy as np

from .base import (
    lib,
    ffi,
    NULL,
    NoValue,
    _check,
    _error_codes,
    _get_bin_op,
    _get_select_op,
    _build_range,
    GxB_INDEX_MAX,
)
from . import types
from .scalar import Scalar
from .semiring import current_semiring, Semiring
from .binaryop import current_accum, current_binop, Accum
from .monoid import current_monoid, Monoid
from . import descriptor
from .descriptor import Descriptor, T1, current_desc


__all__ = ["Vector"]
__pdoc__ = {"Vector.__init__": False}


class Vector:
    """GraphBLAS Sparse Vector

    This is a high-level wrapper around the low-level GrB_Vector type.

    A Vector supports many possible operations according to the
    GraphBLAS API.  Many of those operations have overloaded
    operators.

    Operator | Description | Default
    --- | --- | ---
    v @    A | Vector Vector Multiplication | type default PLUS_TIMES semiring
    v @=   A | In-place Vector Vector Multiplication | type default PLUS_TIMES semiring
    v \\|  w | Vector Union | type default SECOND combiner
    v \\|= w | In-place Vector Union | type default SECOND combiner
    v &    w | Vector Intersection | type default SECOND combiner
    v &=   w | In-place Vector Intersection | type default SECOND combiner
    v +    w | Vector Element-Wise Union | type default PLUS combiner
    v +=   w | In-place Vector Element-Wise Union | type default PLUS combiner
    v -    w | Vector Element-Wise Union | type default MINUS combiner
    v -=   w | In-place Vector Element-Wise Union | type default MINUS combiner
    v *    w | Vector Element-Wise Intersection | type default TIMES combiner
    v *=   w | In-place Vector Element-Wise Intersection | type default TIMES combiner
    v /    w | Vector Element-Wise Intersection | type default DIV combiner
    v /=   w | In-place Vector Element-Wise Intersection | type default DIV combiner
    v ==   w | Compare Element-Wise Union | type default EQ operator
    v !=   w | Compare Element-Wise Union | type default NE operator
    v <    w | Compare Element-Wise Union | type default LT operator
    v >    w | Compare Element-Wise Union | type default GT operator
    v <=   w | Compare Element-Wise Union | type default LE operator
    v >=   w | Compare Element-Wise Union | type default GE operator

    Note that all the above operator syntax is mearly sugar over
    various combinations of calling `Matrix.mxv`, `Vector.vxm`,
    `Vector.eadd`, and `Vector.emult`.

    """

    __slots__ = ("_vector", "type", "_keep_alives")

    def _check(self, res):
        if res != lib.GrB_SUCCESS:
            error_string = ffi.new("char**")
            lib.GrB_Vector_error(error_string, self._vector[0])
            raise _error_codes[res](ffi.string(error_string[0]))

    def __init__(self, vec, typ=None):
        if typ is None:
            new_type = ffi.new("GrB_Type*")
            self._check(lib.GxB_Vector_type(new_type, vec[0]))

            typ = types._gb_type_to_type(new_type[0])

        self._vector = vec
        self.type = typ
        self._keep_alives = weakref.WeakKeyDictionary()

    def __del__(self):
        self._check(lib.GrB_Vector_free(self._vector))

    def __len__(self):
        return self.nvals

    def __iter__(self):
        nvals = self.nvals
        _nvals = ffi.new("GrB_Index[1]", [nvals])
        I = ffi.new("GrB_Index[%s]" % nvals)
        X = ffi.new("%s[%s]" % (self.type._c_type, nvals))
        self._check(self.type._Vector_extractTuples(I, X, _nvals, self._vector[0]))
        return zip(I, X)

    def __getattr__(self, name):
        """Look up operators as attributes for the given object."""
        return partial(getattr(self.type, name), self)

    @property
    def indices(self):
        """cdata array of vector indexes.

        >>> v = Vector.from_1_to_n(3)
        >>> list(v.indices)
        [0, 1, 2]

        """
        nvals = self.nvals
        _nvals = ffi.new("GrB_Index[1]", [nvals])
        I = ffi.new("GrB_Index[%s]" % nvals)
        X = NULL
        self._check(self.type._Vector_extractTuples(I, X, _nvals, self._vector[0]))
        return I

    @property
    def I(self):
        """Iterator over for `Vector.indices`.

        >>> v = Vector.from_1_to_n(3)
        >>> list(v.I)
        [0, 1, 2]

        """
        return iter(self.indices)

    @property
    def npI(self):
        """numpy array over `Vector.indices`.

        >>> v = Vector.from_1_to_n(3)
        >>> v.npI
        array([0, 1, 2], dtype=uint64)

        """
        return np.frombuffer(ffi.buffer(self.indices), dtype=np.uint64)

    @property
    def vals(self):
        """Iterator of vector values.

        >>> v = Vector.from_1_to_n(3)
        >>> list(v.vals)
        [1, 2, 3]

        """
        nvals = self.nvals
        _nvals = ffi.new("GrB_Index[1]", [nvals])
        I = NULL
        V = ffi.new("%s[%s]" % (self.type._c_type, nvals))
        self._check(self.type._Vector_extractTuples(I, V, _nvals, self._vector[0]))
        return V

    @property
    def V(self):
        """Iterator over for `Vector.vals`.

        >>> v = Vector.from_1_to_n(3)
        >>> list(v.V)
        [1, 2, 3]

        """
        return iter(self.vals)

    @property
    def npV(self):
        """numpy array over `Vector.vals`.

        >>> v = Vector.from_1_to_n(3)
        >>> v.npV
        array([1, 2, 3])

        """
        return np.frombuffer(ffi.buffer(self.vals), dtype=self.type._numpy_t)

    def all(self, other, op):
        """Do all elements in self compare True with op to other?

        >>> from . import INT64
        >>> M = Vector.from_lists([0, 1, 2], [1, 2, 3])
        >>> N = Vector.from_lists([0, 1, 2], [1, 2, 3])
        >>> O = Vector.from_lists([0, 1], [1, 2])
        >>> P = Vector.from_lists([0, 1], [1, 2], size=3)
        >>> Q = Vector.from_lists([0, 1, 3], [1, 2, 3])
        >>> assert M.all(N, INT64.eq)
        >>> assert not M.all(N, INT64.gt)
        >>> assert not M.all(O, INT64.eq)
        >>> assert not M.all(P, INT64.eq)
        >>> assert not M.all(Q, INT64.eq)

        """
        if self.size != other.size:
            return False
        if self.nvals != other.nvals:
            return False
        C = self.emult(other, op, cast=types.BOOL)
        if C.nvals != self.nvals:  # pragma: nocover
            return False
        return C.reduce_bool(types.BOOL.land_monoid)

    def iseq(self, other, eq_op=None):
        """Compare two vectors for equality.

        Note to be confused with the `==` operator which does
        element-wise comparison and returns a `Vector`.

        >>> v = Vector.from_lists([0,1], [1, 1])
        >>> w = Vector.from_lists([0,1], [1, 1])
        >>> x = Vector.from_lists([0,1], [1.0, 1.0])
        >>> v.iseq(w)
        True

        >>> v.iseq(w, eq_op=types.UINT64.GE)
        True

        >>> v.iseq(x)
        False
        """
        if self.type != other.type:
            return False
        if eq_op is None:
            eq_op = self.type.EQ
        return self.all(other, eq_op)

    def isne(self, other):
        """Compare two vectors for inequality.
        Note to be confused with the `==` operator which does
        element-wise comparison and returns a `Vector`.

        >>> v = Vector.from_lists([0,1], [1, 1])
        >>> w = Vector.from_lists([0,1], [1, 1])
        >>> v.isne(w)
        False

        """
        return not self.iseq(other)

    @classmethod
    def sparse(cls, typ, size=None, fill=None, mask=None):
        """Create an empty Vector from the given type.  If `size` is not
        specified it defaults to `pygraphblas.GxB_INDEX_MAX`.

        >>> v = Vector.sparse(types.INT64, 3)
        >>> v
        <Vector(INT64 size: 3, nvals: 0)>
        >>> v.size
        3
        >>> v = Vector.sparse(types.INT64)
        >>> v
        <Vector(INT64, nvals: 0)>
        >>> v.size == lib.GxB_INDEX_MAX
        True

        >>> v[42] = True
        >>> w = Vector.sparse(types.INT64, fill=42, mask=v)
        >>> list(w)
        [(42, 42)]

        If no `fill` is provided, the `type.default_zero` is used:

        >>> w = Vector.sparse(types.INT64, mask=v)
        >>> list(w)
        [(42, 0)]
        """
        if size is None:
            size = GxB_INDEX_MAX
        new_vec = ffi.new("GrB_Vector*")
        _check(lib.GrB_Vector_new(new_vec, typ._gb_type, size))
        m = cls(new_vec, typ)
        if mask is not None:
            if fill is None:
                fill = m.type.default_zero
            m.assign_scalar(fill, mask=mask)
        return m

    @classmethod
    def random(
        cls,
        typ,
        nvals,
        size=lib.GxB_INDEX_MAX,
        make_pattern=False,
        seed=None,
    ):  # pragma: nocover
        """ """
        V = Vector.sparse(typ, size)
        if seed is not None:
            random.seed(seed)
        if V.size == 0:
            nvals = 0
        if typ is types.BOOL:
            f = partial(random.randint, 0, 1)
        if typ is types.UINT8:
            f = partial(random.randint, 0, (2 ** 8) - 1)
        if typ is types.UINT16:
            f = partial(random.randint, 0, (2 ** 16) - 1)
        if typ is types.UINT32:
            f = partial(random.randint, 0, (2 ** 32) - 1)
        if typ is types.UINT64:
            f = partial(random.randint, 0, (2 ** 64) - 1)
        if typ is types.INT8:
            f = partial(random.randint, (-(2 ** 7)) + 1, (2 ** 7) - 1)
        if typ is types.INT16:
            f = partial(random.randint, (-(2 ** 15)) + 1, (2 ** 15) - 1)
        if typ is types.INT32:
            f = partial(random.randint, (-(2 ** 31)) + 1, (2 ** 31) - 1)
        if typ is types.INT64:
            f = partial(random.randint, (-(2 ** 63)) + 1, (2 ** 63) - 1)
        if typ in (types.FP32, types.FP64):
            f = random.random
        if typ in (types.FC32, types.FC64):
            f = lambda: complex(random.random(), random.random())
        for i in range(nvals):
            i = random.randint(0, V.size - 1)
            V[i] = f()
        return V

    @classmethod
    def from_lists(cls, I, V, size=None, typ=None):
        """Create a new vector from the given lists of indices and values.  If
        size is not provided, it is computed from the max values of
        the provides size indices.

        If the second argument is a scalar value, an "iso" vector is
        created where all values equal that scalar.

        >>> v = Vector.from_lists([0, 1, 2], [1, 2, 3])
        >>> w = Vector.from_lists([0, 1, 2], True)
        >>> assert not v.iseq(w)
        >>> assert v.pattern().iseq(w)
        """
        if isinstance(V, (bool, int, float)):
            V = [V] * len(I)
        assert len(I) == len(V)
        assert len(I) > 0  # must be non empty
        if not size:
            size = max(I) + 1
        # TODO option to use ffi and GrB_Vector_build
        if typ is None:
            typ = types._gb_from_type(type(V[0]))
        m = cls.sparse(typ, size)
        for i, v in zip(I, V):
            m[i] = v
        return m

    @classmethod
    def from_list(cls, I):
        """Create a new dense vector from the given lists of values."""
        size = len(I)
        assert size > 0
        # TODO use ffi and GrB_Vector_build
        m = cls.sparse(types._gb_from_type(type(I[0])), size)
        for i, v in enumerate(I):
            m[i] = v
        return m

    @classmethod
    def from_1_to_n(cls, n):
        """Generate a vector from 1 to n.

        >>> v = Vector.from_1_to_n(3)
        >>> print(v)
        0| 1
        1| 2
        2| 3
        """
        v = cls.sparse(types.INT64, n)
        for i in range(n):
            v[i] = i + 1
        return v

    def dup(self):
        """Create an duplicate Vector from the given argument.

        >>> v = Vector.from_1_to_n(3)
        >>> w = v.dup()
        >>> w is not v
        True
        >>> w.iseq(v)
        True
        >>> print(w)
        0| 1
        1| 2
        2| 3
        """
        new_vec = ffi.new("GrB_Vector*")
        self._check(lib.GrB_Vector_dup(new_vec, self._vector[0]))
        return self.__class__(new_vec, self.type)

    @property
    def hyper_switch(self):  # pragma: nocover
        """Get the hyper_switch threshold. (See SuiteSparse User Guide)"""
        switch = ffi.new("double*")
        self._check(
            lib.GxB_Vector_Option_get(self._vector[0], lib.GxB_HYPER_SWITCH, switch)
        )
        return switch[0]

    @hyper_switch.setter
    def hyper_switch(self, switch):  # pragma: nocover
        """Set the hyper_switch threshold. (See SuiteSparse User Guide)"""
        switch = ffi.cast("double", switch)
        self._check(
            lib.GxB_Vector_Option_set(self._vector[0], lib.GxB_HYPER_SWITCH, switch)
        )

    @property
    def sparsity(self):  # pragma: nocover
        """Get Vector sparsity control. (See SuiteSparse User Guide)"""
        sparsity = ffi.new("int*")
        self._check(
            lib.GxB_Vector_Option_get(
                self._vector[0], lib.GxB_SPARSITY_CONTROL, sparsity
            )
        )
        return sparsity[0]

    @sparsity.setter
    def sparsity(self, sparsity):  # pragma: nocover
        """Set Vector sparsity control. (See SuiteSparse User Guide)"""
        sparsity = ffi.cast("int", sparsity)
        self._check(
            lib.GxB_Vector_Option_set(
                self._vector[0], lib.GxB_SPARSITY_CONTROL, sparsity
            )
        )

    @property
    def sparsity_status(self):  # pragma: nocover
        """Get Vector sparsity status. (See SuiteSparse User Guide)"""
        status = ffi.new("int*")
        self._check(
            lib.GxB_Vector_Option_get(self._vector[0], lib.GxB_SPARSITY_STATUS, status)
        )
        return status[0]

    @classmethod
    def dense(cls, typ, size=None, fill=None):
        """Return a dense vector of `typ` and `size`.  If `fill` is provided,
        use that value otherwise use `self.type.default_zero`

        >>> print(Vector.dense(types.FP32, 3))
        0|0.0
        1|0.0
        2|0.0
        >>> print(Vector.dense(types.FP32, 3, fill=42.0))
        0|42.0
        1|42.0
        2|42.0

        """
        v = cls.sparse(typ, size)
        if fill is None:
            fill = v.type.default_zero
        v[:] = fill
        return v

    @classmethod
    def iso(cls, value, size=GxB_INDEX_MAX):
        """Build an "iso" vector from a scalar value.

        This is similar to `Vector.dense` but infers the type of the
        new Matrix from the provided vbalue.

        >>> v = Vector.iso(3)
        >>> assert v[42] == 3
        """
        typ = types._gb_from_type(type(value))
        return cls.dense(typ, size, value)

    def to_lists(self):
        """Extract the indices and values of the Vector as 2 lists.

        >>> Vector.from_1_to_n(3).to_lists()
        [[0, 1, 2], [1, 2, 3]]

        """
        I = ffi.new("GrB_Index[]", self.nvals)
        V = self.type._ffi.new(self.type._c_type + "[]", self.nvals)
        n = ffi.new("GrB_Index*")
        n[0] = self.nvals
        self._check(self.type._Vector_extractTuples(I, V, n, self._vector[0]))
        return [list(I), list(map(self.type._to_value, V))]

    def to_arrays(self):
        """Return as python `array` objects.

        >>> Vector.from_1_to_n(3).to_arrays()
        (array('L', [0, 1, 2]), array('q', [1, 2, 3]))

        """
        if self.type._typecode is None:
            raise TypeError("This matrix has no array typecode.")
        nvals = self.nvals
        _nvals = ffi.new("GrB_Index[1]", [nvals])
        I = ffi.new("GrB_Index[%s]" % nvals)
        X = self.type._ffi.new("%s[%s]" % (self.type._c_type, nvals))
        self._check(self.type._Vector_extractTuples(I, X, _nvals, self._vector[0]))
        return array("L", I), array(self.type._typecode, X)

    @property
    def size(self):
        """Return the size of the vector.

        >>> Vector.from_1_to_n(3).size
        3

        """
        n = ffi.new("GrB_Index*")
        self._check(lib.GrB_Vector_size(n, self._vector[0]))
        return n[0]

    @property
    def nvals(self):
        """Return the number of values in the vector.

        >>> v = Vector.from_1_to_n(3)
        >>> v.nvals
        3
        >>> v.clear()
        >>> v.nvals
        0

        """
        n = ffi.new("GrB_Index*")
        self._check(lib.GrB_Vector_nvals(n, self._vector[0]))
        return n[0]

    @property
    def memory_usage(self):
        """Returns the memory usage of the Vector.

        >>> v = Vector.from_lists([0, 1, 2], [1, 2, 0])
        >>> assert v.memory_usage > 0
        """
        n = ffi.new("size_t*")
        self._check(lib.GxB_Vector_memoryUsage(n, self._vector[0]))
        return n[0]

    @property
    def gb_type(self):
        """Return the GraphBLAS low-level type object of the Vector."""
        return self.type._gb_type

    def _full(self):
        B = self.__class__.sparse(self.type, self.size)

        self._check(
            self.type._Vector_assignScalar(
                B._vector[0], NULL, NULL, self.type.default_one, lib.GrB_ALL, 0, NULL
            )
        )
        return self.eadd(B, self.type.FIRST)

    def _compare(self, other, op, strop):
        C = self.__class__.sparse(types.BOOL, self.size)
        if isinstance(other, (bool, int, float, complex)):
            if op(other, 0):
                B = self.__class__.dup(self)
                B[:] = other
                self.emult(B, strop, out=C)
                return C
            else:
                self.select(strop, other).apply(types.BOOL.ONE, out=C)
                return C
        elif isinstance(other, Vector):
            A = self._full()
            B = other._full()
            A.emult(B, strop, out=C)
            return C
        else:
            raise NotImplementedError

    def __gt__(self, other):
        return self._compare(other, operator.gt, ">")

    def __lt__(self, other):
        return self._compare(other, operator.lt, "<")

    def __ge__(self, other):
        return self._compare(other, operator.ge, ">=")

    def __le__(self, other):
        return self._compare(other, operator.le, "<=")

    def __eq__(self, other):
        return self._compare(other, operator.eq, "==")

    def __ne__(self, other):
        return self._compare(other, operator.ne, "!=")

    def eadd(
        self,
        other,
        add_op=None,
        cast=None,
        out=None,
        mask=None,
        accum=None,
        desc=None,
    ):
        """Element-wise addition with other vector.

        Element-wise addition applies a binary operator element-wise
        on two vectors `v` and `w`, for all entries that appear in the
        set union of the patterns of `A` and `B`.

        The only difference between element-wise multiplication and
        addition is the pattern of the result, and what happens to
        entries outside the intersection. With multiplication the
        pattern of T is the intersection; with addition it is the set
        union. Entries outside the set intersection are dropped for
        multiplication, and kept for addition; in both cases the
        operator is only applied to those (and only those) entries in
        the intersection. Any binary operator can be used
        interchangeably for either operation.

        >>> I = [0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6]
        >>> V = list(range(len(I)))
        >>> v = Vector.from_lists(I, V, 7)

        >>> w = Vector.from_lists(
        ...    [0, 1, 4, 6],
        ...    [9, 1, 4, 7], 7)

        >>> print(v.eadd(w))
        0|10
        1| 4
        2| 4
        3| 6
        4|11
        5| 8
        6|18

        This can also be accomplished with the `+` operators:

        >>> print(v + w)
        0|10
        1| 4
        2| 4
        3| 6
        4|11
        5| 8
        6|18

        The combining operator used can be provided either as a
        context manager or passed to `mxv` as the `add_op` argument.

        >>> with types.INT64.MIN:
        ...     print(v + w)
        0| 1
        1| 1
        2| 4
        3| 6
        4| 4
        5| 8
        6| 7

        You can provide a monoid for the operation:

        >>> print(v.eadd(w, v.type.min_monoid))
        0| 1
        1| 1
        2| 4
        3| 6
        4| 4
        5| 8
        6| 7

        Or you can use a semiring:

        >>> print(v.eadd(w, v.type.min_plus))
        0| 1
        1| 1
        2| 4
        3| 6
        4| 4
        5| 8
        6| 7

        The following operators default to use `eadd`:

        Operator | Description | Default
        --- | --- | ---
        v \\|  w | Vector Union | type default SECOND combiner
        v \\|= w | In-place Vector Union | type default SECOND combiner
        v +    w | Vector Element-Wise Union | type default PLUS combiner
        v +=   w | In-place Vector Element-Wise Union | type default PLUS combiner
        v -    w | Vector Element-Wise Union | type default MINUS combiner
        v -=   w | In-place Vector Element-Wise Union | type default MINUS combiner

        """
        func = lib.GrB_Vector_eWiseAdd_BinaryOp
        if add_op is None:
            add_op = current_binop.get(NULL)
        elif isinstance(add_op, Monoid):
            func = lib.GrB_Vector_eWiseAdd_Monoid
        elif isinstance(add_op, Semiring):
            func = lib.GrB_Vector_eWiseAdd_Semiring

        mask, accum, desc = self._get_args(mask, accum, desc)

        if out is None:
            typ = cast or types.promote(self.type, other.type)
            _out = ffi.new("GrB_Vector*")
            self._check(lib.GrB_Vector_new(_out, typ._gb_type, self.size))
            out = self.__class__(_out, typ)

        if add_op is NULL:
            add_op = out.type._default_addop()

        add_op = add_op.get_op()
        self._check(
            func(
                out._vector[0],
                mask,
                accum,
                add_op,
                self._vector[0],
                other._vector[0],
                desc,
            )
        )
        return out

    def emult(
        self,
        other,
        mult_op=None,
        cast=None,
        out=None,
        mask=None,
        accum=None,
        desc=None,
    ):
        """Element-wise multiplication with other vector.

        Element-wise multiplication applies a binary operator
        element-wise on two vectors A and B, for all entries that
        appear in the set intersection of the patterns of A and B.

        The only difference between element-wise multiplication and
        addition is the pattern of the result, and what happens to
        entries outside the intersection. With multiplication the
        pattern of T is the intersection; with addition it is the set
        union. Entries outside the set intersection are dropped for
        multiplication, and kept for addition; in both cases the
        operator is only applied to those (and only those) entries in
        the intersection. Any binary operator can be used
        interchangeably for either operation.

        >>> I = [0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6]
        >>> V = list(range(len(I)))
        >>> v = Vector.from_lists(I, V, 7)

        >>> w = Vector.from_lists(
        ...    [0, 1, 4, 6],
        ...    [9, 1, 4, 7], 7)

        >>> print(v.emult(w))
        0| 9
        1| 3
        2|
        3|
        4|28
        5|
        6|77

        This can also be accomplished with the `+` operators:

        >>> print(v * w)
        0| 9
        1| 3
        2|
        3|
        4|28
        5|
        6|77

        The combining operator used can be provided either as a
        context manager or passed to `mxv` as the `add_op` argument.

        >>> with types.INT64.MAX:
        ...     print(v * w)
        0| 9
        1| 3
        2|
        3|
        4|28
        5|
        6|77

        """
        if mult_op is None:
            mult_op = current_binop.get(NULL)
        elif isinstance(mult_op, str):
            mult_op = _get_bin_op(mult_op, self.type)

        mask, accum, desc = self._get_args(mask, accum, desc)
        if out is None:
            typ = cast or types.promote(self.type, other.type)
            _out = ffi.new("GrB_Vector*")
            self._check(lib.GrB_Vector_new(_out, typ._gb_type, self.size))
            out = self.__class__(_out, typ)

        if mult_op is NULL:
            mult_op = out.type._default_multop()

        mult_op = mult_op.get_op()
        self._check(
            lib.GrB_Vector_eWiseMult_BinaryOp(
                out._vector[0],
                mask,
                accum,
                mult_op,
                self._vector[0],
                other._vector[0],
                desc,
            )
        )
        return out

    def vxm(
        self,
        other,
        semiring=None,
        cast=None,
        out=None,
        mask=None,
        accum=None,
        desc=None,
    ):
        """Vector-Matrix multiply.


        Multiply this row vector by `other` matrix "on the left".  For
        column matrix/vector multiplication "on the right" see
        `Matrix.mxv`.

        `vxm` can also be called directly or with the `@` operator:

        >>> from . import Matrix, INT64
        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [1, 2, 3])
        >>> v = Vector.from_lists([0, 1, 2], [2, 3, 4])
        >>> o = v.vxm(M)
        >>> print(o)
        0|12
        1| 2
        2| 6
        >>> o = v @ M
        >>> print(o)
        0|12
        1| 2
        2| 6

        By default, `mxv` and `@` create a new result matrix of the
        correct type and dimensions if one is not provided.  If you
        want to provide your own matrix to put the result in, you can
        pass it in the `out` parameter.  This is useful for
        accumulating results into a single matrix with minimal
        copying.  This is also supported by the `@=` syntax:

        >>> o = v.dup()
        >>> v.vxm(M, accum=INT64.plus, out=o) is o
        True
        >>> print(o)
        0|14
        1| 5
        2|10
        >>> o = v.dup()
        >>> with Accum(INT64.min):
        ...     o @= M
        >>> print(o)
        0| 2
        1| 2
        2| 4

        The default semiring depends on the infered result type.  In
        the case of numbers, the default semiring is `PLUS_TIMES`.  In
        the case of type `BOOL`, it is `BOOL.lor_land`.

        >>> o = v.vxm(M, semiring=INT64.min_plus)
        >>> print(o)
        0| 7
        1| 3
        2| 5

        An explicit semiring can be passed to the method or provided
        with a context manager:

        >>> with INT64.min_plus:
        ...     o = v @ M
        >>> print(o)
        0| 7
        1| 3
        2| 5

        Or the semiring can be accessed via an attribute on the
        vector:

        >>> o = v.min_plus(M)
        >>> print(o)
        0| 7
        1| 3
        2| 5

        Descriptors and accumulators can also be provided as an
        argument or a context manager:

        >>> o = v.vxm(M, desc=descriptor.T0)
        >>> print(o)
        0|12
        1| 2
        2| 6
        >>> with descriptor.T0:
        ...     o = v @ M
        >>> print(o)
        0|12
        1| 2
        2| 6
        >>> del o[1]
        >>> o = v.vxm(M, mask=o)
        >>> print(o)
        0|12
        1|
        2| 6

        """

        if semiring is None:
            semiring = current_semiring.get(NULL)

        if out is None:
            new_dimension = other.nrows if T1 in (desc or ()) else other.ncols
            if semiring is not NULL:
                typ = semiring.ztype
            else:
                typ = cast or types.promote(self.type, other.type)
            out = Vector.sparse(typ, new_dimension)
        else:
            typ = out.type

        if semiring is NULL:
            semiring = out.type._default_semiring()

        semiring = semiring.get_op()
        mask, accum, desc = self._get_args(mask, accum, desc)
        self._check(
            lib.GrB_vxm(
                out._vector[0],
                mask,
                accum,
                semiring,
                self._vector[0],
                other._matrix[0],
                desc,
            )
        )
        return out

    def __matmul__(self, other):
        return self.vxm(other)

    def __imatmul__(self, other):
        return self.vxm(other, out=self)

    def __and__(self, other):
        return self.emult(other)

    def __iand__(self, other):
        return self.emult(other, out=self)

    def __or__(self, other):
        return self.eadd(other)

    def __ior__(self, other):
        return self.eadd(other, out=self)

    def __add__(self, other):
        if not isinstance(other, Vector):
            return self.apply_second(self.type.PLUS, other)
        return self.eadd(other)

    def __radd__(self, other):
        if not isinstance(other, Vector):
            return self.apply_first(other, self.type.PLUS)
        return other.eadd(self)  # pragma: nocover

    def __iadd__(self, other):
        if not isinstance(other, Vector):
            return self.apply_second(self.type.PLUS, other, out=self)
        return self.eadd(other, out=self)

    def __sub__(self, other):
        if not isinstance(other, Vector):
            return self.apply_second(self.type.MINUS, other)
        return self.eadd(other, self.type.MINUS)

    def __rsub__(self, other):
        if not isinstance(other, Vector):
            return self.apply_first(other, self.type.MINUS)
        return other.eadd(self, self.type.MINUS)  # pragma: nocover

    def __isub__(self, other):
        if not isinstance(other, Vector):
            return self.apply_second(self.type.MINUS, other)
        return other.eadd(self, self.type.MINUS, out=self)

    def __mul__(self, other):
        if not isinstance(other, Vector):
            return self.apply_second(self.type.TIMES, other)
        return self.emult(other, self.type.TIMES)

    def __rmul__(self, other):
        if not isinstance(other, Vector):
            return self.apply_first(other, self.type.TIMES)
        return other.emult(self, add_op=self.type.TIMES)  # pragma: nocover

    def __imul__(self, other):
        if not isinstance(other, Vector):
            return self.apply_second(self.type.TIMES, other, out=self)
        return other.emult(self, self.type.TIMES, out=self)

    def __truediv__(self, other):
        if not isinstance(other, Vector):
            return self.apply_second(self.type.DIV, other)
        return self.emult(other, self.type.DIV)

    def __rtruediv__(self, other):
        if not isinstance(other, Vector):
            return self.apply_first(other, self.type.DIV)
        return other.emult(self, self.type.DIV)  # pragma: nocover

    def __itruediv__(self, other):
        if not isinstance(other, Vector):
            return self.apply_second(self.type.DIV, other, out=self)
        return other.emult(self, self.type.DIV, out=self)

    def __invert__(self):
        return self.apply(self.type.MINV)

    def __neg__(self):
        return self.apply(self.type.AINV)

    def __abs__(self):
        return self.apply(self.type.ABS)

    def clear(self):
        """Clear this vector removing all entries."""
        self._check(lib.GrB_Vector_clear(self._vector[0]))

    def resize(self, size=lib.GxB_INDEX_MAX):
        """Resize the vector.  If the dimensions decrease, entries that fall
        outside the resized vector are deleted.

        >>> v = Vector.dense(types.UINT8, 2)
        >>> v.resize(3)
        >>> print(v)
        0| 0
        1| 0
        2|

        """
        self._check(lib.GrB_Vector_resize(self._vector[0], size))

    def _get_args(self, mask=None, accum=None, desc=None):
        if accum is None:
            accum = current_accum.get(NULL)

        if accum is not NULL:
            accum = accum.get_op()

        if desc is None:
            desc = current_desc.get(NULL)

        if desc is not NULL:
            desc = desc.get_desc()

        if mask is None:
            mask = NULL
        # else:
        #     if desc is None:
        #         desc = S

        if isinstance(mask, Vector):
            mask = mask._vector[0]
        return mask, accum, desc

    def reduce(self, mon=None, accum=None, desc=None):
        """Do a scalar reduce based on this object's type:

        >>> V = Vector.random(types.UINT8, 10, 3, seed=42)
        >>> V.reduce()
        114

        >>> V = Vector.random(types.FP32, 10, 3, seed=42)
        >>> V.reduce()
        0.9517456293106079

        >>> V = Vector.random(types.UINT8, 10, 3, seed=42)
        >>> V.reduce(V.type.min_monoid)
        13

        >>> V = Vector.random(types.BOOL, 10, 3, seed=42)
        >>> V.reduce()
        False

        """
        if mon is None:
            if self.type is types.BOOL:
                mon = current_monoid.get(getattr(self.type, "lor_monoid"))
            else:
                mon = current_monoid.get(getattr(self.type, "plus_monoid"))
        mon = mon.get_op()
        mask, accum, desc = self._get_args(None, accum, desc)
        result = ffi.new(self.type._c_type + "*")
        self._check(self.type._Vector_reduce(result, accum, mon, self._vector[0], desc))
        return result[0]

    def reduce_bool(self, mon=None, mask=None, accum=None, desc=None):
        """Reduce vector to a boolean.

        >>> v = Vector.from_lists([0, 1], [True, False])
        >>> v.reduce_bool()
        True
        >>> v[0] = False
        >>> v.reduce_bool()
        False
        >>> v[1] = True
        >>> v.reduce_bool(types.BOOL.LAND_MONOID)
        False

        """
        if mon is None:
            mon = current_monoid.get(types.BOOL.LOR_MONOID)
        mon = mon.get_op()
        mask, accum, desc = self._get_args(mask, accum, desc)
        result = ffi.new("_Bool*")
        self._check(
            lib.GrB_Vector_reduce_BOOL(result, accum, mon, self._vector[0], desc)
        )
        return result[0]

    def reduce_int(self, mon=None, mask=None, accum=None, desc=None):
        """Reduce vector to a integer.

        >>> v = Vector.from_lists([0, 1], [1, 1])
        >>> v.reduce_int()
        2
        >>> v[0] = 0
        >>> v.reduce_int()
        1
        >>> v[1] = 2
        >>> v.reduce_int(types.INT64.MIN_MONOID)
        0

        """
        if mon is None:
            mon = current_monoid.get(types.INT64.PLUS_MONOID)
        mon = mon.get_op()
        mask, accum, desc = self._get_args(mask, accum, desc)
        result = ffi.new("int64_t*")
        self._check(
            lib.GrB_Vector_reduce_INT64(result, accum, mon, self._vector[0], desc)
        )
        return result[0]

    def reduce_float(self, mon=None, mask=None, accum=None, desc=None):
        """Reduce vector to a float.

        >>> v = Vector.from_lists([0, 1], [1.2, 1.1])
        >>> v.reduce_float()
        2.3
        >>> v[0] = 0
        >>> v.reduce_float()
        1.1
        >>> v[1] = 2.2
        >>> v.reduce_float(types.FP64.MIN_MONOID)
        0.0

        """
        if mon is None:
            mon = current_monoid.get(types.FP64.PLUS_MONOID)
        mon = mon.get_op()
        mask, accum, desc = self._get_args(mask, accum, desc)
        result = ffi.new("double*")
        self._check(
            lib.GrB_Vector_reduce_FP64(result, accum, mon, self._vector[0], desc)
        )
        return result[0]

    def max(self):
        """Return the max of the vector.

        >>> M = Vector.from_lists([0, 1, 2], [False, False, False])
        >>> M.max()
        False
        >>> M = Vector.from_lists([0, 1, 2], [False, False, True])
        >>> M.max()
        True
        >>> M = Vector.from_lists([0, 1, 2], [-42, 0, 149])
        >>> M.max()
        149
        >>> M = Vector.from_lists([0, 1, 2], [-42.0, 0.0, 149.0])
        >>> M.max()
        149.0
        >>> M = Vector.from_lists([0], [1j])
        >>> M.max()
        Traceback (most recent call last):
        ...
        TypeError: Un-maxable type
        """
        if self.type == types.BOOL:
            return self.reduce_bool(self.type.LOR_MONOID)
        if self.type in types._int_types:
            return self.reduce_int(self.type.MAX_MONOID)
        if self.type in types._float_types:
            return self.reduce_float(self.type.MAX_MONOID)
        raise TypeError("Un-maxable type")

    def min(self):
        """Return the min of the vector.

        >>> M = Vector.from_lists([0, 1, 2], [True, True, True])
        >>> M.min()
        True
        >>> M = Vector.from_lists([0, 1, 2], [False, True, True])
        >>> M.min()
        False
        >>> M = Vector.from_lists([0, 1, 2], [-42, 0, 149])
        >>> M.min()
        -42
        >>> M = Vector.from_lists([0, 1, 2], [-42.0, 0.0, 149.0])
        >>> M.min()
        -42.0
        >>> M = Vector.from_lists([0], [1j])
        >>> M.min()
        Traceback (most recent call last):
        ...
        TypeError: Un-minable type
        """
        if self.type == types.BOOL:
            return self.reduce_bool(self.type.LAND_MONOID)
        if self.type in types._int_types:
            return self.reduce_int(self.type.MIN_MONOID)
        if self.type in types._float_types:
            return self.reduce_float(self.type.MIN_MONOID)
        raise TypeError("Un-minable type")

    def apply(self, op, out=None, mask=None, accum=None, desc=None):
        """Apply Unary op to vector elements.
        >>> from . import UINT64
        >>> v = Vector.from_lists([0,1], [1, 1])
        >>> print(v.apply(UINT64.ainv))
        0|-1
        1|-1

        Unary operators can also be accessed by atribute name on
        vectors they are applied to:

        >>> print(v.ainv())
        0|-1
        1|-1

        """
        if out is None:
            out = Vector.sparse(self.type, self.size)

        op = op.get_op()
        mask, accum, desc = self._get_args(mask, accum, desc)
        self._check(
            lib.GrB_Vector_apply(out._vector[0], mask, accum, op, self._vector[0], desc)
        )
        return out

    def apply_first(self, first, op, out=None, mask=None, accum=None, desc=None):
        """Apply a binary operator to the entries in a vector, binding the first input
        to a scalar first.


        >>> v = Vector.from_lists([0,1], [1, 1])
        >>> print(v.apply_first(3, types.UINT64.PLUS))
        0| 4
        1| 4
        >>> w = Vector.sparse(v.type, v.size)
        >>> v.apply_first(3, types.UINT64.PLUS, out=w) is w
        True

        """
        if out is None:
            out = self.__class__.sparse(self.type, self.size)

        op = op.get_op()
        mask, accum, desc = self._get_args(mask, accum, desc)
        if isinstance(first, Scalar):
            f = lib.GxB_Vector_apply_BinaryOp1st
            first = first._scalar[0]
        else:
            f = self.type._Vector_apply_BinaryOp1st
        self._check(f(out._vector[0], mask, accum, op, first, self._vector[0], desc))
        return out

    def apply_second(self, op, second, out=None, mask=None, accum=None, desc=None):
        """Apply a binary operator to the entries in a vector, binding the second input
        to a scalar second.

        >>> v = Vector.from_lists([0,1], [1, 1])
        >>> print(v.apply_second(types.UINT64.PLUS, 3))
        0| 4
        1| 4
        >>> w = Vector.sparse(v.type, v.size)
        >>> v.apply_second(types.UINT64.PLUS, 3, out=w) is w
        True
        >>> u = Vector.from_lists([0,1], [1.1, 2.2])
        >>> u.apply_second(u.type.TIMES, 3.3, out=u) is u
        True
        >>> u = Vector.from_lists([0,1], [1.1, 2.2])
        >>> print(u * 3)
        0|3.3
        1|6.6
        >>> x = Vector.from_lists([0,1], [1.1, 2.2])
        >>> x *= 3.0
        >>> print(x)
        0|3.3
        1|6.6

        """
        if out is None:
            out = self.__class__.sparse(self.type, self.size)

        op = op.get_op()
        mask, accum, desc = self._get_args(mask, accum, desc)
        if isinstance(second, Scalar):
            f = lib.GxB_Vector_apply_BinaryOp2nd
            second = second._scalar[0]
        else:
            f = self.type._Vector_apply_BinaryOp2nd

        self._check(f(out._vector[0], mask, accum, op, self._vector[0], second, desc))
        return out

    def select(self, op, thunk=None, out=None, mask=None, accum=None, desc=None):
        """Select elements that match the given select operation condition.
        See `Matrix.select` for possible operators.

        >>> v = Vector.from_lists([0,1], [1, 0])
        >>> print(v.select('>', 0))
        0| 1
        1|

        >>> w = Vector.sparse(types.UINT8, 2)
        >>> v.select('>', 0, out=w) is w
        True

        `min` and `max` selectors can be shortcuts for selecting all
        elements that equal the min or max reduction of all elements.

        >>> print(v.select('min'))
        0|
        1| 0
        >>> print(v.select('max'))
        0| 1
        1|

        """
        if out is None:
            out = Vector.sparse(self.type, self.size)
        if isinstance(op, str):
            if op == "min":
                op = lib.GxB_EQ_THUNK
                thunk = self.min()
            elif op == "max":
                op = lib.GxB_EQ_THUNK
                thunk = self.max()
            else:
                op = _get_select_op(op)

        if thunk is None:
            thunk = NULL
        if isinstance(thunk, (bool, int, float, complex)):
            thunk = Scalar.from_value(thunk)
        if isinstance(thunk, Scalar):
            self._keep_alives[self._vector] = thunk
            thunk = thunk._scalar[0]

        mask, accum, desc = self._get_args(mask, accum, desc)
        self._check(
            lib.GxB_Vector_select(
                out._vector[0], mask, accum, op, self._vector[0], thunk, desc
            )
        )
        return out

    def pattern(self, typ=types.BOOL):
        """Return the pattern of the vector, this is a boolean Vector where
        every present value in this vector is set to True.

        """
        result = Vector.sparse(typ, self.size)
        self.apply(types.BOOL.ONE, out=result)
        return result

    @property
    def S(self):
        """Return the vector "structure".  This is the same as calling
        `Vector.pattern()` with no arguments.

        >>> v = Vector.from_lists([0, 1, 2], [1, 2, 3])
        >>> assert v.S == v.pattern()

        """
        return self.pattern()

    def nonzero(self):
        """Select vector of nonzero entries."""
        return self.select(lib.GxB_NONZERO)

    def __setitem__(self, index, value):
        if isinstance(index, int):
            val = self.type._from_value(value)
            self._check(self.type._Vector_setElement(self._vector[0], val, index))
            return

        if isinstance(index, slice):
            if isinstance(value, Vector):
                self.assign(value, index)
                return
            if isinstance(value, (bool, int, float, complex)):
                self.assign_scalar(value, index)
                return

        if isinstance(index, Vector):
            mask = index._vector[0]
            index = slice(None, None, None)
            if isinstance(value, Vector):
                self.assign(value, index, mask=mask)
                return
            self.assign_scalar(value, index, mask=mask)
            return
        raise TypeError("Unknown index")

    def assign(self, value, index=None, mask=None, accum=None, desc=None):
        """Assign vector to vector.

        >>> v = Vector.sparse(types.INT8, 3)
        >>> w = Vector.from_1_to_n(3)
        >>> v[:] = w
        >>> print(v)
        0| 1
        1| 2
        2| 3

        If the index is another vector it is used as an assignment
        mask:

        >>> v.clear()
        >>> m = Vector.sparse(types.BOOL, 3)
        >>> m[1] = True
        >>> v[m] = w
        >>> print(v)
        0|
        1| 2
        2|
        >>> v.clear()
        >>> m = Vector.sparse(types.BOOL, 3)
        >>> m[1] = True
        >>> v[m] = 3
        >>> print(v)
        0|
        1| 3
        2|

        """
        mask, accum, desc = self._get_args(mask, accum, desc)
        I, ni, size = _build_range(index, self.size - 1)
        self._check(
            lib.GrB_Vector_assign(
                self._vector[0], mask, accum, value._vector[0], I, ni, desc
            )
        )

    def assign_scalar(self, value, index=None, mask=None, accum=None, desc=None):
        """Assign scalar to vector.

        >>> v = Vector.sparse(types.INT8, 3)
        >>> v[:] = 2
        >>> print(v)
        0| 2
        1| 2
        2| 2

        If the index is another vector it is used as an assignment
        mask:

        >>> v.clear()
        >>> m = Vector.sparse(types.BOOL, 3)
        >>> m[1] = True
        >>> v[m] = 3
        >>> print(v)
        0|
        1| 3
        2|

        """
        mask, accum, desc = self._get_args(mask, accum, desc)
        scalar_type = types._gb_from_type(type(value))
        I, ni, size = _build_range(index, self.size - 1)
        self._check(
            scalar_type._Vector_assignScalar(
                self._vector[0], mask, accum, value, I, ni, desc
            )
        )

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.extract_element(index)
        else:
            return self.extract(index)

    def __delitem__(self, index):
        if not isinstance(index, int):
            raise TypeError(
                "__delitem__ currently only supports single element removal"
            )
        self._check(lib.GrB_Vector_removeElement(self._vector[0], index))

    def extract_element(self, index):
        """Extract element from vector."""
        result = self.type._ffi.new(self.type._ptr)
        self._check(
            self.type._Vector_extractElement(
                result, self._vector[0], ffi.cast("GrB_Index", index)
            )
        )
        return self.type._to_value(result[0])

    def extract(self, index, mask=None, accum=None, desc=None):
        """Extract subvector from vector."""
        mask, accum, desc = self._get_args(mask, accum, desc)
        if isinstance(index, Vector):
            mask = index._vector[0]
            index = slice(None, None, None)
        I, ni, size = _build_range(index, self.size - 1)
        if size is None:
            size = self.size
        result = Vector.sparse(self.type, size)
        self._check(
            lib.GrB_Vector_extract(
                result._vector[0], mask, accum, self._vector[0], I, ni, desc
            )
        )
        return result

    def __contains__(self, index):
        try:
            v = self[index]
            return True
        except NoValue:
            return False

    def get(self, i, default=None):
        """Get element at `i` or return `default` if not present.


        >>> M = Vector.from_lists([1, 2], [42, 149])
        >>> M.get(1)
        42
        >>> M.get(0) is None
        True
        >>> M.get(0, 'foo')
        'foo'

        """
        try:
            return self[i]
        except NoValue:
            return default

    def wait(self):
        """Wait for vector to complete."""
        self._check(lib.GrB_Vector_wait(self._vector))

    def to_string(self, format_string="{:>%s}", width=2, prec=3, empty_char=""):
        """Return string representation of vector."""
        format_string = format_string % width
        result = ""
        for row in range(self.size):
            value = self.get(row, empty_char)
            result += str(row) + "|"
            result += format_string.format(
                self.type.format_value(value, width, prec)
            ).rstrip()
            if row < self.size - 1:
                result += "\n"
        return result

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        tname = self.type.__name__
        if self.size == lib.GxB_INDEX_MAX:
            return f"<Vector({tname}, nvals: {self.nvals})>"

        return f"<Vector({tname} size: {self.size}, nvals: {self.nvals})>"

    def print(self, level=2, name="A", f=sys.stdout):  # pragma: nocover
        """Print the matrix using `GxB_Matrix_fprint()`, by default to
        `sys.stdout`..

        Level 1: Short description
        Level 2: Short list, short numbers
        Level 3: Long list, short number
        Level 4: Short list, long numbers
        Level 5: Long list, long numbers

        """
        self._check(
            lib.GxB_Vector_fprint(self._vector[0], bytes(name, "utf8"), level, f)
        )
