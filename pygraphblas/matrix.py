"""High level wrapper around GraphBLAS Matrices.

"""
import sys
import weakref
import operator
from random import randint
from array import array

from .base import (
    lib,
    ffi,
    NULL,
    NoValue,
    _check,
    _error_codes,
    _build_range,
    _get_select_op,
    _get_bin_op,
    GxB_INDEX_MAX,
)

from . import types
from .vector import Vector
from .scalar import Scalar
from .semiring import current_semiring
from .binaryop import current_accum, current_binop, Accum
from .monoid import current_monoid
from .descriptor import Descriptor, Default, T0, current_desc
from . import descriptor
from .descriptor import Descriptor, Default, T0, current_desc
from .gviz import draw_graph, draw_matrix

__all__ = ["Matrix"]
__pdoc__ = {"Matrix.__init__": False}


class Matrix:
    """GraphBLAS Sparse Matrix

    This is a high-level wrapper around the GrB_Matrix C type using
    the [cffi](https://cffi.readthedocs.io/en/latest/) library.

    A Matrix supports many possible operations according to the
    GraphBLAS API.  Many of those operations have overloaded
    operators.

    Operator | Description | Default
    --- | --- | ---
    A @    B | Matrix Matrix Multiplication | type default PLUS_TIMES semiring
    v @    A | Vector Matrix Multiplication | type default PLUS_TIMES semiring
    A @    v | Matrix Vector Multiplication | type default PLUS_TIMES semiring
    A @=   B | In-place Matrix Matrix Multiplication | type default PLUS_TIMES semiring
    v @=   A | In-place Vector Matrix Multiplication | type default PLUS_TIMES semiring
    A @=   v | In-place Matrix Vector Multiplication | type default PLUS_TIMES semiring
    A \\|  B | Matrix Union | type default SECOND combiner
    A \\|= B | In-place Matrix Union | type default SECOND combiner
    A &    B | Matrix Intersection | type default SECOND combiner
    A &=   B | In-place Matrix Intersection | type default SECOND combiner
    A +    B | Matrix Element-Wise Union | type default PLUS combiner
    A +=   B | In-place Matrix Element-Wise Union | type default PLUS combiner
    A -    B | Matrix Element-Wise Union | type default MINUS combiner
    A -=   B | In-place Matrix Element-Wise Union | type default MINUS combiner
    A *    B | Matrix Element-Wise Intersection | type default TIMES combiner
    A *=   B | In-place Matrix Element-Wise Intersection | type default TIMES combiner
    A /    B | Matrix Element-Wise Intersection | type default DIV combiner
    A /=   B | In-place Matrix Element-Wise Intersection | type default DIV combiner
    A ==   B | Compare Element-Wise Union | type default EQ operator
    A !=   B | Compare Element-Wise Union | type default NE operator
    A <    B | Compare Element-Wise Union | type default LT operator
    A >    B | Compare Element-Wise Union | type default GT operator
    A <=   B | Compare Element-Wise Union | type default LE operator
    A >=   B | Compare Element-Wise Union | type default GE operator

    Note that all the above operator syntax is mearly sugar over
    various combinations of calling `Matrix.mxm`, `Matrix.mxv`,
    `pygraphblas.Vector.vxm`, `Matrix.eadd`, and `Matrix.emult`.

    """

    __slots__ = ("_matrix", "type", "_funcs", "_keep_alives")

    def _check(self, res, raise_no_val=False):
        if res != lib.GrB_SUCCESS:
            if raise_no_val and res == lib.GrB_NO_VALUE:
                raise KeyError  # pragma: nocover

            error_string = ffi.new("char**")
            lib.GrB_Matrix_error(error_string, self._matrix[0])
            raise _error_codes[res](ffi.string(error_string[0]))

    def __init__(self, matrix, typ=None):
        if typ is None:
            new_type = ffi.new("GrB_Type*")
            self._check(lib.GxB_Matrix_type(new_type, matrix[0]))

            typ = types.gb_type_to_type(new_type[0])

        self._matrix = matrix
        self.type = typ
        """The type of the Matrix. 

        >>> M = Matrix.sparse(types.INT8)
        >>> M.type == types.INT8
        True
        """
        self._keep_alives = weakref.WeakKeyDictionary()

    def __del__(self):
        self._check(lib.GrB_Matrix_free(self._matrix))

    @classmethod
    def sparse(cls, typ, nrows=None, ncols=None):
        """Create an empty sparse Matrix from the given type.  The dimensions
        can be specified with `nrows` and `ncols`.  If no dimensions
        are specified, they default to `GxB_INDEX_MAX`.

        >>> m = Matrix.sparse(types.UINT8)
        >>> m.nrows == lib.GxB_INDEX_MAX
        True
        >>> m.ncols == lib.GxB_INDEX_MAX
        True
        >>> m.nvals == 0
        True

        Optional row and column dimension bounds can be provided to
        the method:

        >>> m = Matrix.sparse(types.UINT8, 10, 10)
        >>> m.nrows == 10
        True
        >>> m.ncols == 10
        True
        >>> m.nvals == 0
        True

        """
        if nrows is None:
            nrows = GxB_INDEX_MAX
        if ncols is None:
            ncols = GxB_INDEX_MAX
        new_mat = ffi.new("GrB_Matrix*")
        _check(lib.GrB_Matrix_new(new_mat, typ.gb_type, nrows, ncols))
        m = cls(new_mat, typ)
        return m

    @classmethod
    def dense(cls, typ, nrows, ncols, fill=None, sparsity=None):
        """Return a dense Matrix nrows by ncols.

        If `sparsity` is provided it is used for the sparsity of the
        new matrix See the [SuiteSparse User
        Guide](https://raw.githubusercontent.com/DrTimothyAldenDavis/GraphBLAS/stable/Doc/GraphBLAS_UserGuide.pdf)
        for details.

        >>> M = Matrix.dense(types.UINT8, 3, 3)
        >>> print(M)
              0  1  2
          0|  0  0  0|  0
          1|  0  0  0|  1
          2|  0  0  0|  2
              0  1  2

        If a `fill` value is present, use that, otherwise use the
        `zero` attribte of the given type.

        >>> M = Matrix.dense(types.UINT8, 3, 3, fill=1)
        >>> print(M)
              0  1  2
          0|  1  1  1|  0
          1|  1  1  1|  1
          2|  1  1  1|  2
              0  1  2

        """
        assert nrows > 0 and ncols > 0, "dense matrix must be at least 1x1"
        m = cls.sparse(typ, nrows, ncols)
        if sparsity is not None:
            m.sparsity = sparsity
        if fill is None:
            fill = m.type.zero
        m[:, :] = fill
        return m

    @classmethod
    def from_lists(cls, I, J, V, nrows=None, ncols=None, typ=None):
        """Create a new matrix from the given lists of row indices, column
        indices, and values.  If nrows or ncols are not provided, they
        are computed from the max values of the provides row and
        column indices lists.

        >>> I = [0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6]
        >>> J = [1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4]
        >>> V = [True for _ in range(len(I))]
        >>> M = Matrix.from_lists(I, J, V)
        >>> print(M)
              0  1  2  3  4  5  6
          0|     t     t         |  0
          1|              t     t|  1
          2|                 t   |  2
          3|  t     t            |  3
          4|                 t   |  4
          5|        t            |  5
          6|        t  t  t      |  6
              0  1  2  3  4  5  6
        >>> from pygraphblas.gviz import draw_graph
        >>> draw_graph(M, filename='/docs/imgs/Matrix_from_lists')
        <graphviz.dot.Digraph object at ...>

        ![Matrix_from_lists.png](../imgs/Matrix_from_lists.png)

        """
        assert len(I) == len(J) == len(V)
        if not nrows:
            nrows = max(I) + 1
        if not ncols:
            ncols = max(J) + 1
        # TODO use ffi and GrB_Matrix_build
        if typ is None:
            typ = types._gb_from_type(type(V[0]))
        m = cls.sparse(typ, nrows, ncols)
        for i, j, v in zip(I, J, V):
            m[i, j] = v
        return m

    @classmethod
    def from_mm(cls, mm_file, typ):
        """Create a new matrix by reading a Matrix Market file.

        >>> with open('/docs/test_mm.mm', 'r') as f:
        ...     M = Matrix.from_mm(f, types.INT8)
        >>> print(M)
              0  1  2  3  4  5  6
          0|     0     1         |  0
          1|              2     3|  1
          2|                 4   |  2
          3|  5     6            |  3
          4|                 7   |  4
          5|        8            |  5
          6|        9 10 11      |  6
              0  1  2  3  4  5  6

        """
        m = ffi.new("GrB_Matrix*")
        i = cls(m, typ)
        _check(lib.LAGraph_mmread(m, mm_file))
        return i

    @classmethod
    def from_tsv(cls, tsv_file, typ, nrows, ncols):
        """Create a new matrix by reading a tab separated value file."""
        m = ffi.new("GrB_Matrix*")
        i = cls(m, typ)
        _check(lib.LAGraph_tsvread(m, tsv_file, typ.gb_type, nrows, ncols))
        return i

    @classmethod
    def from_binfile(cls, bin_file):
        """Create a new matrix by reading a SuiteSparse specific binary file.

        >>> M = Matrix.from_binfile(bytes('/docs/test_binfile.grb', 'utf8'))
        >>> print(M)
              0  1  2  3  4  5  6
          0|     0     1         |  0
          1|              2     3|  1
          2|                 4   |  2
          3|  5     6            |  3
          4|                 7   |  4
          5|        8            |  5
          6|        9 10 11      |  6
              0  1  2  3  4  5  6

        """
        m = ffi.new("GrB_Matrix*")
        _check(lib.LAGraph_binread(m, bin_file))
        return cls(m)

    @classmethod
    def random(
        cls,
        typ,
        nrows,
        ncols,
        nvals,
        make_pattern=False,
        make_symmetric=False,
        make_skew_symmetric=False,
        make_hermitian=True,
        no_diagonal=False,
        seed=None,
    ):
        """Create a new random Matrix of the given type, number of rows,
        columns and values.  Other flags set additional properties the
        matrix will hold.

        >>> M = Matrix.random(types.UINT8, 5, 5, 20,
        ...                   make_symmetric=True, no_diagonal=True, seed=42)
        >>> draw_graph(M, filename='/docs/imgs/Matrix_random')
        <graphviz.dot.Digraph object at ...>

        ![Matrix_random.png](../imgs/Matrix_random.png)

        """
        result = ffi.new("GrB_Matrix*")
        i = cls(result, typ)
        fseed = ffi.new("uint64_t*")
        if seed is None:
            seed = randint(0, sys.maxsize)
        fseed[0] = seed
        _check(
            lib.LAGraph_random(
                result,
                typ.gb_type,
                nrows,
                ncols,
                nvals,
                make_pattern,
                make_symmetric,
                make_skew_symmetric,
                make_hermitian,
                no_diagonal,
                fseed,
            )
        )
        return i

    @classmethod
    def identity(cls, typ, nrows, one=None):
        """Return a new square identity Matrix of nrows with diagonal set to
        one.

        If one is None, use the default typ one value.

        >>> M = Matrix.identity(types.UINT8, 3, one=42)
        >>> print(M)
              0  1  2
          0| 42      |  0
          1|    42   |  1
          2|       42|  2
              0  1  2

        """
        result = cls.sparse(typ, nrows, nrows)
        if one is None:
            one = result.type.one
        for i in range(nrows):
            result[i, i] = one
        return result

    @property
    def gb_type(self):
        """Return the GraphBLAS low-level type object of the Matrix.  This is
        only used if interacting with the low level API.

        >>> M = Matrix.sparse(types.INT8)
        >>> M.gb_type == lib.GrB_INT8
        True

        """
        new_type = ffi.new("GrB_Type*")
        self._check(lib.GxB_Matrix_type(new_type, self._matrix[0]))
        return new_type[0]

    @property
    def nrows(self):
        """Return the number of Matrix rows.

        >>> M = Matrix.sparse(types.UINT8, 3, 3)
        >>> M.nrows
        3

        """
        n = ffi.new("GrB_Index*")
        self._check(lib.GrB_Matrix_nrows(n, self._matrix[0]))
        return n[0]

    @property
    def ncols(self):
        """Return the number of Matrix columns.

        >>> M = Matrix.sparse(types.UINT8, 3, 3)
        >>> M.ncols
        3

        """
        n = ffi.new("GrB_Index*")
        self._check(lib.GrB_Matrix_ncols(n, self._matrix[0]))
        return n[0]

    @property
    def shape(self):
        """Numpy-like description of matrix shape as 2-tuple (nrows, ncols).

        >>> M = Matrix.sparse(types.UINT8, 3, 3)
        >>> M.shape
        (3, 3)

        """
        return (self.nrows, self.ncols)

    @property
    def square(self):
        """True if Matrix is square, else False.

        >>> M = Matrix.sparse(types.UINT8, 3, 3)
        >>> M.square
        True
        >>> M = Matrix.sparse(types.UINT8, 3, 4)
        >>> M.square
        False

        """
        return self.nrows == self.ncols

    @property
    def nvals(self):
        """Return the number of values stored in the Matrix.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 1492])
        >>> M.nvals
        3

        """
        n = ffi.new("GrB_Index*")
        self._check(lib.GrB_Matrix_nvals(n, self._matrix[0]))
        return n[0]

    @property
    def T(self):
        """Compute transpose of the Matrix.  See `Matrix.transpose`.

        Note: This property can be expensive, if you need the
        transpose more than once, consider storing this in a local
        variable.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 1492])
        >>> MT = M.T
        >>> MT.iseq(M.transpose())
        True

        """
        return self.transpose()

    @property
    def M(self):
        """Return the structural "mask" pattern of this matrix.  See
        `pattern()`.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 142])
        >>> print(M)
              0  1  2
          0|    42   |  0
          1|      314|  1
          2|142      |  2
              0  1  2
        >>> print(M.M)
              0  1  2
          0|     t   |  0
          1|        t|  1
          2|  t      |  2
              0  1  2

        """
        return self.pattern()

    def dup(self, clear=False):
        """Create an duplicate Matrix.

        If `clear` is true return an empty duplicate.

        >>> A = Matrix.sparse(types.UINT8)
        >>> A[1,1] = 42
        >>> B = A.dup()
        >>> B[1,1]
        42
        >>> B is not A
        True
        >>> C = A.dup(True)
        >>> assert not C

        """
        if clear:
            return self.__class__.sparse(self.type, self.nrows, self.ncols)
        new_mat = ffi.new("GrB_Matrix*")
        self._check(lib.GrB_Matrix_dup(new_mat, self._matrix[0]))
        return self.__class__(new_mat, self.type)

    @property
    def hyper_switch(self):
        """Get the hyper_switch threshold. (See SuiteSparse User Guide)"""
        switch = ffi.new("double*")
        self._check(
            lib.GxB_Matrix_Option_get(self._matrix[0], lib.GxB_HYPER_SWITCH, switch)
        )
        return switch[0]

    @hyper_switch.setter
    def hyper_switch(self, switch):
        """Set the hyper_switch threshold. (See SuiteSparse User Guide)"""
        switch = ffi.cast("double", switch)
        self._check(
            lib.GxB_Matrix_Option_set(self._matrix[0], lib.GxB_HYPER_SWITCH, switch)
        )

    @property
    def format(self):
        """Get Matrix format. (See SuiteSparse User Guide)"""
        format = ffi.new("GxB_Format_Value*")
        self._check(lib.GxB_Matrix_Option_get(self._matrix[0], lib.GxB_FORMAT, format))
        return format[0]

    @format.setter
    def format(self, format):
        """Set Matrix format. (See SuiteSparse User Guide)"""
        format = ffi.cast("GxB_Format_Value", format)
        self._check(lib.GxB_Matrix_Option_set(self._matrix[0], lib.GxB_FORMAT, format))

    @property
    def sparsity(self):
        """Get Matrix sparsity control. (See SuiteSparse User Guide)"""
        sparsity = ffi.new("int*")
        self._check(
            lib.GxB_Matrix_Option_get(
                self._matrix[0], lib.GxB_SPARSITY_CONTROL, sparsity
            )
        )
        return sparsity[0]

    @sparsity.setter
    def sparsity(self, sparsity):
        """Set Matrix sparsity control. (See SuiteSparse User Guide)"""
        sparsity = ffi.cast("int", sparsity)
        self._check(
            lib.GxB_Matrix_Option_set(
                self._matrix[0], lib.GxB_SPARSITY_CONTROL, sparsity
            )
        )

    @property
    def sparsity_status(self):
        """Set Matrix sparsity status. (See SuiteSparse User Guide)"""
        status = ffi.new("int*")
        self._check(
            lib.GxB_Matrix_Option_get(self._matrix[0], lib.GxB_SPARSITY_STATUS, status)
        )
        return status[0]

    def pattern(self, typ=types.BOOL, out=None):
        """Return the pattern of the matrix where every present value in this
        matrix is set to identity value for the provided type which
        defaults to BOOL.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 142])
        >>> print(M)
              0  1  2
          0|    42   |  0
          1|      314|  1
          2|142      |  2
              0  1  2
        >>> P = M.pattern()
        >>> print(P)
              0  1  2
          0|     t   |  0
          1|        t|  1
          2|  t      |  2
              0  1  2

        Pre-constructed matrix can be passed as the `out` parameter:

        >>> C = Matrix.dense(types.BOOL, 3, 3)
        >>> P = M.pattern(out=C)
        >>> print(C)
              0  1  2
          0|     t   |  0
          1|        t|  1
          2|  t      |  2
              0  1  2

        """

        if out is None:
            out = Matrix.sparse(typ, self.nrows, self.ncols)
        return self.apply(typ.ONE, out=out)

    def to_mm(self, fileobj):
        """Write this matrix to a file using the Matrix Market format."""
        self._check(lib.LAGraph_mmwrite(self._matrix[0], fileobj))

    def to_binfile(self, filename, comments=""):
        """Write this matrix using custom SuiteSparse binary format."""
        self._check(
            lib.LAGraph_binwrite(self._matrix, filename, bytes(comments, "utf8"))
        )

    def to_lists(self):
        """Extract the rows, columns and values of the Matrix as 3 lists.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 1492])
        >>> M.to_lists()
        [[0, 1, 2], [1, 2, 0], [42, 314, 1492]]

        """
        I = ffi.new("GrB_Index[%s]" % self.nvals)
        J = ffi.new("GrB_Index[%s]" % self.nvals)
        V = self.type.ffi.new(self.type.C + "[%s]" % self.nvals)
        n = ffi.new("GrB_Index*")
        n[0] = self.nvals
        self._check(self.type._Matrix_extractTuples(I, J, V, n, self._matrix[0]))
        return [list(I), list(J), list(map(self.type.to_value, V))]

    def clear(self):
        """Clear the matrix.  This does not change the size but removes all
        values.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 1492])
        >>> M.nvals == 3
        True
        >>> M.clear()
        >>> print(M)
              0  1  2
          0|         |  0
          1|         |  1
          2|         |  2
              0  1  2

        """
        self._check(lib.GrB_Matrix_clear(self._matrix[0]))

    def resize(self, nrows, ncols):
        """Resize the matrix.  If the dimensions decrease, entries that fall
        outside the resized matrix are deleted.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 149])
        >>> M.shape
        (3, 3)
        >>> M.resize(10, 10)
        >>> print(M)
              0  1  2  3  4  5  6  7  8  9
          0|    42                        |  0
          1|      314                     |  1
          2|149                           |  2
          3|                              |  3
          4|                              |  4
          5|                              |  5
          6|                              |  6
          7|                              |  7
          8|                              |  8
          9|                              |  9
              0  1  2  3  4  5  6  7  8  9

        """
        self._check(lib.GrB_Matrix_resize(self._matrix[0], nrows, ncols))

    def transpose(self, cast=None, out=None, mask=None, accum=None, desc=Default):
        """Return Transpose of this matrix.

        This function can serve multiple interesting purposes
        including typecasting.  See the [SuiteSparse User
        Guide](https://raw.githubusercontent.com/DrTimothyAldenDavis/GraphBLAS/stable/Doc/GraphBLAS_UserGuide.pdf)

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 149])
        >>> print(M)
              0  1  2
          0|    42   |  0
          1|      314|  1
          2|149      |  2
              0  1  2

        >>> MT = M.transpose()
        >>> print(MT)
              0  1  2
          0|      149|  0
          1| 42      |  1
          2|   314   |  2
              0  1  2

        >>> MT = M.transpose(cast=types.BOOL, desc=descriptor.T0)
        >>> print(MT)
              0  1  2
          0|     t   |  0
          1|        t|  1
          2|  t      |  2
              0  1  2

        >>> N = M.dup(True)
        >>> MT = M.transpose(desc=descriptor.T0, out=N)
        >>> print(MT)
              0  1  2
          0|    42   |  0
          1|      314|  1
          2|149      |  2
              0  1  2

        """
        if out is None:
            new_dimensions = (
                (self.nrows, self.ncols) if T0 in desc else (self.ncols, self.nrows)
            )
            _out = ffi.new("GrB_Matrix*")
            if cast is not None:
                typ = cast
            else:
                typ = self.type
            self._check(lib.GrB_Matrix_new(_out, typ.gb_type, *new_dimensions))
            out = self.__class__(_out, typ)
        mask, accum, desc = self._get_args(mask, accum, desc)
        self._check(
            lib.GrB_transpose(out._matrix[0], mask, accum, self._matrix[0], desc)
        )
        return out

    def cast(self, cast, out=None):
        """Cast this matrix to the provided type.  If out is not provided, a
        new matrix is of the cast type is created.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 149])
        >>> print(M)
              0  1  2
          0|    42   |  0
          1|      314|  1
          2|149      |  2
              0  1  2
        >>> N = M.cast(types.FP32)
        >>> print(N.to_string(width=5, prec=4))
                  0    1    2
            0|      42.0     |  0
            1|          314.0|  1
            2|149.0          |  2
                  0    1    2

        >>> N = M.cast(types.FP64)
        >>> print(N.to_string(width=5, prec=4))
                  0    1    2
            0|      42.0     |  0
            1|          314.0|  1
            2|149.0          |  2
                  0    1    2

        """
        return self.transpose(cast, out, desc=T0)

    def eadd(
        self,
        other,
        add_op=None,
        cast=None,
        out=None,
        mask=None,
        accum=None,
        desc=Default,
    ):
        """Element-wise addition with other matrix

        Element-wise addition takes the set union of the patterns of A
        and B and applies a binary operator for all entries that
        appear in the set intersection of the patterns of A and B.
        The default operators is the `PLUS` binary operator of the
        output type.

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
        >>> J = [1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4]
        >>> V = list(range(len(I)))
        >>> A = Matrix.from_lists(I, J, V, 7, 7)
        >>> draw_graph(A, filename='/docs/imgs/Matrix_eadd_A')
        <graphviz.dot.Digraph object at ...>

        ![Matrix_eadd_A.png](../imgs/Matrix_eadd_A.png)

        >>> B = Matrix.from_lists(
        ...    [0, 1, 4, 6],
        ...    [1, 3, 5, 5],
        ...    [9, 1, 4, 7], 7, 7)
        >>> draw_graph(B, filename='/docs/imgs/Matrix_eadd_B')
        <graphviz.dot.Digraph object at ...>

        ![Matrix_eadd_B.png](../imgs/Matrix_eadd_B.png)

        >>> draw_graph(A.eadd(B), filename='/docs/imgs/Matrix_eadd_C')
        <graphviz.dot.Digraph object at ...>
        >>> print(A.eadd(B))
              0  1  2  3  4  5  6
          0|     9     1         |  0
          1|           1  2     3|  1
          2|                 4   |  2
          3|  5     6            |  3
          4|                11   |  4
          5|        8            |  5
          6|        9 10 11  7   |  6
              0  1  2  3  4  5  6

        ![Matrix_eadd_C.png](../imgs/Matrix_eadd_C.png)

        This can also be accomplished with the `+` operators:

        >>> print(A + B)
              0  1  2  3  4  5  6
          0|     9     1         |  0
          1|           1  2     3|  1
          2|                 4   |  2
          3|  5     6            |  3
          4|                11   |  4
          5|        8            |  5
          6|        9 10 11  7   |  6
              0  1  2  3  4  5  6

        The combining operator used can be provided either as a
        context manager or passed to `mxv` as the `add_op` argument.

        >>> with types.INT64.MIN:
        ...     print(A + B)
              0  1  2  3  4  5  6
          0|     0     1         |  0
          1|           1  2     3|  1
          2|                 4   |  2
          3|  5     6            |  3
          4|                 4   |  4
          5|        8            |  5
          6|        9 10 11  7   |  6
              0  1  2  3  4  5  6

        The following operators default to use `eadd`:

        Operator | Description | Default
        --- | --- | ---
        A \\|  B | Matrix Union | type default SECOND combiner
        A \\|= B | In-place Matrix Union | type default SECOND combiner
        A +    B | Matrix Element-Wise Union | type default PLUS combiner
        A +=   B | In-place Matrix Element-Wise Union | type default PLUS combiner
        A -    B | Matrix Element-Wise Union | type default MINUS combiner
        A -=   B | In-place Matrix Element-Wise Union | type default MINUS combiner

        """
        if add_op is None:
            add_op = current_binop.get(NULL)
        elif isinstance(add_op, str):
            add_op = _get_bin_op(add_op, self.type)

        mask, accum, desc = self._get_args(mask, accum, desc)
        if out is None:
            typ = cast or types.promote(self.type, other.type)
            _out = ffi.new("GrB_Matrix*")
            self._check(lib.GrB_Matrix_new(_out, typ.gb_type, self.nrows, self.ncols))
            out = Matrix(_out, typ)

        if add_op is NULL:
            add_op = out.type._default_addop()
        add_op = add_op.get_binaryop(self.type, other.type)
        self._check(
            lib.GrB_Matrix_eWiseAdd_BinaryOp(
                out._matrix[0],
                mask,
                accum,
                add_op,
                self._matrix[0],
                other._matrix[0],
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
        desc=Default,
    ):
        """Element-wise multiplication with other matrix.

        Element-wise multiplication applies a binary operator
        element-wise on two matrices A and B, for all entries that
        appear in the set intersection of the patterns of A and B.
        Other operators other than addition can be used.

        The pattern of the result of the element-wise multiplication
        is exactly this set intersection. Entries in A but not B, or
        visa versa, do not appear in the result.

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
        >>> J = [1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4]
        >>> V = list(range(len(I)))
        >>> A = Matrix.from_lists(I, J, V, 7, 7)
        >>> draw_graph(A, filename='/docs/imgs/Matrix_emult_A')
        <graphviz.dot.Digraph object at ...>

        ![Matrix_emult_A.png](../imgs/Matrix_emult_A.png)

        >>> B = Matrix.from_lists(
        ...    [0, 1, 1, 6, 6],
        ...    [1, 4, 6, 3, 5],
        ...    [9, 1, 4, 7, 11], 7, 7)
        >>> draw_graph(B, filename='/docs/imgs/Matrix_emult_B')
        <graphviz.dot.Digraph object at ...>

        ![Matrix_emult_B.png](../imgs/Matrix_emult_B.png)

        >>> draw_graph(A.emult(B), filename='/docs/imgs/Matrix_emult_C')
        <graphviz.dot.Digraph object at ...>
        >>> print(A.emult(B))
              0  1  2  3  4  5  6
          0|     0               |  0
          1|              2    12|  1
          2|                     |  2
          3|                     |  3
          4|                     |  4
          5|                     |  5
          6|          70         |  6
              0  1  2  3  4  5  6

        ![Matrix_emult_C.png](../imgs/Matrix_emult_C.png)

        This can also be accomplished with the `+` operators:

        >>> print(A * B)
              0  1  2  3  4  5  6
          0|     0               |  0
          1|              2    12|  1
          2|                     |  2
          3|                     |  3
          4|                     |  4
          5|                     |  5
          6|          70         |  6
              0  1  2  3  4  5  6

        The combining operator used can be provided either as a
        context manager or passed to `mxv` as the `add_op` argument.

        >>> with types.INT64.MIN:
        ...     print(A * B)
              0  1  2  3  4  5  6
          0|     0               |  0
          1|              1     3|  1
          2|                     |  2
          3|                     |  3
          4|                     |  4
          5|                     |  5
          6|           7         |  6
              0  1  2  3  4  5  6

        The following operators default to using `emult`:

        Operator | Description | Default
        --- | --- | ---
        A &    B | Matrix Intersection | type default SECOND combiner
        A &=   B | In-place Matrix Intersection | type default SECOND combiner
        A *    B | Matrix Element-Wise Intersection | type default TIMES combiner
        A *=   B | In-place Matrix Element-Wise Intersection | type default TIMES combiner
        A /    B | Matrix Element-Wise Intersection | type default DIV combiner
        A /=   B | In-place Matrix Element-Wise Intersection | type default DIV combiner

        """
        if mult_op is None:
            mult_op = current_binop.get(NULL)
        elif isinstance(mult_op, str):
            mult_op = _get_bin_op(mult_op, self.type)

        mask, accum, desc = self._get_args(mask, accum, desc)
        if out is None:
            typ = cast or types.promote(self.type, other.type)
            _out = ffi.new("GrB_Matrix*")
            self._check(lib.GrB_Matrix_new(_out, typ.gb_type, self.nrows, self.ncols))
            out = Matrix(_out, typ)

        if mult_op is NULL:
            mult_op = out.type._default_multop()
        mult_op = mult_op.get_binaryop(self.type, other.type)
        self._check(
            lib.GrB_Matrix_eWiseMult_BinaryOp(
                out._matrix[0],
                mask,
                accum,
                mult_op,
                self._matrix[0],
                other._matrix[0],
                desc,
            )
        )
        return out

    def iseq(self, other):
        """Compare two matrices for equality returning True or False.

        Not to be confused with `==` which will return a matrix of
        BOOL values comparing *elements* for equality.

        Uses code from LAGraph_isequal.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 1492])
        >>> N = M.dup()
        >>> M.iseq(N)
        True
        >>> del N[0, 1]
        >>> M.iseq(N)
        False

        """
        result = ffi.new("_Bool*")
        eq_op = self.type.EQ.get_binaryop(self.type, other.type)
        self._check(
            lib.LAGraph_isequal(result, self._matrix[0], other._matrix[0], eq_op)
        )
        return result[0]

    def isne(self, other):
        """Compare two matrices for inequality.  See `Matrix.iseq`."""
        return not self.iseq(other)

    def __iter__(self):
        """Iterate over the (row, col, value) triples of the Matrix.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 1492])
        >>> sorted(list(iter(M)))
        [(0, 1, 42), (1, 2, 314), (2, 0, 1492)]

        """
        nvals = self.nvals
        _nvals = ffi.new("GrB_Index[1]", [nvals])
        I = ffi.new("GrB_Index[%s]" % nvals)
        J = ffi.new("GrB_Index[%s]" % nvals)
        X = self.type.ffi.new("%s[%s]" % (self.type.C, nvals))
        self._check(self.type._Matrix_extractTuples(I, J, X, _nvals, self._matrix[0]))
        return zip(I, J, map(self.type.to_value, X))

    def to_arrays(self):
        """Convert Matrix to tuple of three dense
        [array](https://docs.python.org/3/library/array.html) objects.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 1492])
        >>> M.to_arrays()
        (array('L', [0, 1, 2]), array('L', [1, 2, 0]), array('q', [42, 314, 1492]))

        """
        if self.type.typecode is None:
            raise TypeError("This matrix has no array typecode.")
        nvals = self.nvals
        _nvals = ffi.new("GrB_Index[1]", [nvals])
        I = ffi.new("GrB_Index[%s]" % nvals)
        J = ffi.new("GrB_Index[%s]" % nvals)
        X = self.type.ffi.new("%s[%s]" % (self.type.C, nvals))
        self._check(self.type._Matrix_extractTuples(I, J, X, _nvals, self._matrix[0]))
        return array("L", I), array("L", J), array(self.type.typecode, X)

    @property
    def rows(self):
        """An iterator of row indexes present in the matrix.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 1492])
        >>> list(M.rows)
        [0, 1, 2]

        """
        nvals = self.nvals
        _nvals = ffi.new("GrB_Index[1]", [nvals])
        I = ffi.new("GrB_Index[%s]" % nvals)
        J = NULL
        X = NULL
        self._check(self.type._Matrix_extractTuples(I, J, X, _nvals, self._matrix[0]))
        return iter(I)

    @property
    def cols(self):
        """An iterator of column indexes present in the matrix.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 1492])
        >>> list(M.cols)
        [1, 2, 0]

        """
        nvals = self.nvals
        _nvals = ffi.new("GrB_Index[1]", [nvals])
        I = NULL
        J = ffi.new("GrB_Index[%s]" % nvals)
        X = NULL
        self._check(self.type._Matrix_extractTuples(I, J, X, _nvals, self._matrix[0]))
        return iter(J)

    @property
    def vals(self):
        """An iterator of values present in the matrix.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 1492])
        >>> list(M.vals)
        [42, 314, 1492]

        """
        nvals = self.nvals
        _nvals = ffi.new("GrB_Index[1]", [nvals])
        I = NULL
        J = NULL
        X = self.type.ffi.new("%s[%s]" % (self.type.C, nvals))
        self._check(self.type._Matrix_extractTuples(I, J, X, _nvals, self._matrix[0]))
        return iter(X)

    def __len__(self):
        """Return the number of elements in the Matrix.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 314, 1492])
        >>> len(M)
        3

        """
        return self.nvals

    def __and__(self, other):
        op = current_binop.get(self.type.SECOND)
        return self.emult(other, op)

    def __iand__(self, other):
        op = current_binop.get(self.type.SECOND)
        return self.emult(other, op, out=self)

    def __or__(self, other):
        op = current_binop.get(self.type.SECOND)
        return self.eadd(other, op)

    def __ior__(self, other):
        op = current_binop.get(self.type.SECOND)
        return self.eadd(other, op, out=self)

    def __add__(self, other):
        op = current_binop.get(self.type.PLUS)
        if not isinstance(other, Matrix):
            return self.apply_second(op, other)
        return self.eadd(other, op)

    def __radd__(self, other):
        op = current_binop.get(self.type.PLUS)
        if not isinstance(other, Matrix):
            return self.apply_first(other, op)
        return other.eadd(self, op)  # pragma: nocover

    def __iadd__(self, other):
        op = current_binop.get(self.type.PLUS)
        if not isinstance(other, Matrix):
            return self.apply_second(op, other, out=self)
        return self.eadd(other, op, out=self)

    def __sub__(self, other):
        op = current_binop.get(self.type.MINUS)
        if not isinstance(other, Matrix):
            return self.apply_second(op, other)
        return self.eadd(other, op)

    def __rsub__(self, other):
        op = current_binop.get(self.type.MINUS)
        if not isinstance(other, Matrix):
            return self.apply_first(other, op)
        return other.eadd(self, op)  # pragma: nocover

    def __isub__(self, other):
        op = current_binop.get(self.type.MINUS)
        if not isinstance(other, Matrix):
            return self.apply_second(op, other, out=self)
        return other.eadd(self, op, out=self)

    def __mul__(self, other):
        op = current_binop.get(self.type.TIMES)
        if not isinstance(other, Matrix):
            return self.apply_second(op, other)
        return self.emult(other, op)

    def __rmul__(self, other):
        op = current_binop.get(self.type.TIMES)
        if not isinstance(other, Matrix):
            return self.apply_first(other, op)
        return other.emult(self, op)  # pragma: nocover

    def __imul__(self, other):
        op = current_binop.get(self.type.TIMES)
        if not isinstance(other, Matrix):
            return self.apply_second(op, other)
        return other.emult(self, op, out=self)

    def __truediv__(self, other):
        op = current_binop.get(self.type.DIV)
        if not isinstance(other, Matrix):
            return self.apply_second(op, other)
        return self.emult(other, op)

    def __rtruediv__(self, other):
        op = current_binop.get(self.type.DIV)
        if not isinstance(other, Matrix):
            return self.apply_first(other, op)
        return other.emult(self, op)  # pragma: nocover

    def __itruediv__(self, other):
        op = current_binop.get(self.type.DIV)
        if not isinstance(other, Matrix):
            return self.apply_second(op, other)
        return other.emult(self, op, out=self)

    def __invert__(self):
        return self.apply(self.type.MINV)

    def __neg__(self):
        return self.apply(self.type.AINV)

    def __abs__(self):
        return self.apply(self.type.ABS)

    def __pow__(self, exponent):
        if exponent == 0:
            return self.__class__.identity(self.type, self.nrows)
        if exponent == 1:
            return self
        result = self.dup()
        for i in range(1, exponent):
            result.mxm(self, out=result)
        return result

    def reduce_bool(self, mon=None, mask=None, accum=None, desc=Default):
        """Reduce matrix to a boolean.

        >>> M = Matrix.sparse(types.INT8)
        >>> M.reduce_bool()
        False
        >>> M[0,1] = True
        >>> M.reduce_bool()
        True

        >>> M.reduce_bool(types.BOOL.LOR_MONOID)
        True
        """
        if mon is None:
            mon = current_monoid.get(types.BOOL.LOR_MONOID)
        mon = mon.get_monoid(self.type)
        result = ffi.new("_Bool*")
        mask, accum, desc = self._get_args(mask, accum, desc)
        self._check(
            lib.GrB_Matrix_reduce_BOOL(result, accum, mon, self._matrix[0], desc)
        )
        return result[0]

    def reduce_int(self, mon=None, mask=None, accum=None, desc=Default):
        """Reduce matrix to an integer.

        >>> M = Matrix.sparse(types.INT8)
        >>> M.reduce_int()
        0
        >>> M[0,1] = 42
        >>> M[0,2] = 42
        >>> M.reduce_int()
        84
        >>> M.reduce_int(types.INT8.MIN_MONOID)
        42

        """
        if mon is None:
            mon = current_monoid.get(types.INT64.PLUS_MONOID)
        mon = mon.get_monoid(self.type)
        result = ffi.new("int64_t*")
        mask, accum, desc = self._get_args(mask, accum, desc)
        self._check(
            lib.GrB_Matrix_reduce_INT64(result, accum, mon, self._matrix[0], desc)
        )
        return result[0]

    def reduce_float(self, mon=None, mask=None, accum=None, desc=Default):
        """Reduce matrix to an float.

        >>> M = Matrix.sparse(types.FP32)
        >>> M.reduce_float()
        0.0
        >>> M[0,1] = 42.0
        >>> M[0,2] = 42.0
        >>> M.reduce_float()
        84.0

        """
        if mon is None:
            mon = current_monoid.get(self.type.PLUS_MONOID)
        mon = mon.get_monoid(self.type)
        mask, accum, desc = self._get_args(mask, accum, desc)
        result = ffi.new("double*")
        self._check(
            lib.GrB_Matrix_reduce_FP64(result, accum, mon, self._matrix[0], desc)
        )
        return result[0]

    def reduce_vector(self, mon=None, out=None, mask=None, accum=None, desc=Default):
        """Reduce matrix to a vector.

        >>> M = Matrix.sparse(types.FP32, 3, 3)
        >>> print(M.reduce_vector())
        0|
        1|
        2|
        >>> M[0,1] = 42.0
        >>> M[0,2] = 42.0
        >>> M[2,0] = -42.0
        >>> print(M.reduce_vector())
        0|84.0
        1|
        2|-42.0

        >>> print(M.reduce_vector(types.FP32.MIN_MONOID))
        0|42.0
        1|
        2|-42.0

        >>> v = Vector.sparse(types.FP32, M.nrows)
        >>> print(M.reduce_vector(out=v))
        0|84.0
        1|
        2|-42.0
        """
        if mon is None:
            mon = current_monoid.get(getattr(self.type, "PLUS_MONOID", NULL))
        mon = mon.get_monoid(self.type)
        if out is None:
            out = Vector.sparse(self.type, self.nrows)
        mask, accum, desc = self._get_args(mask, accum, desc)
        self._check(
            lib.GrB_Matrix_reduce_Monoid(
                out._vector[0], mask, accum, mon, self._matrix[0], desc
            )
        )
        return out

    def apply(self, op, out=None, mask=None, accum=None, desc=Default):
        """Apply Unary op to matrix elements.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [-42, 0, 149])
        >>> print(M.apply(types.INT64.ABS))
              0  1  2
          0|    42   |  0
          1|        0|  1
          2|149      |  2
              0  1  2

        >>> print(M.apply(types.INT64.ABS))
              0  1  2
          0|    42   |  0
          1|        0|  1
          2|149      |  2
              0  1  2
        """
        if out is None:
            out = self.__class__.sparse(self.type, self.nrows, self.ncols)

        op = op.get_unaryop(self.type)
        mask, accum, desc = self._get_args(mask, accum, desc)
        self._check(
            lib.GrB_Matrix_apply(out._matrix[0], mask, accum, op, self._matrix[0], desc)
        )
        return out

    def apply_first(self, first, op, out=None, mask=None, accum=None, desc=Default):
        """Apply a binary operator to the entries in a matrix, binding the
        first input to a scalar first.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [-42, 0, 149])
        >>> print(M.apply_first(1, types.INT64.PLUS))
              0  1  2
          0|   -41   |  0
          1|        1|  1
          2|150      |  2
              0  1  2

        >>> N = Matrix.sparse(M.type, M.nrows, M.ncols)
        >>> print(M.apply_first(1, types.INT64.PLUS, out=N))
              0  1  2
          0|   -41   |  0
          1|        1|  1
          2|150      |  2
              0  1  2

        """
        if out is None:
            out = self.__class__.sparse(self.type, self.nrows, self.ncols)
        op = op.get_binaryop(self.type)
        mask, accum, desc = self._get_args(mask, accum, desc)
        if isinstance(first, Scalar):
            f = lib.GxB_Matrix_apply_BinaryOp1st
            first = first._scalar[0]
        else:
            f = self.type._Matrix_apply_BinaryOp1st
        self._check(f(out._matrix[0], mask, accum, op, first, self._matrix[0], desc))
        return out

    def apply_second(self, op, second, out=None, mask=None, accum=None, desc=Default):
        """Apply a binary operator to the entries in a matrix, binding the
        second input to a scalar second.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [-42, 0, 149])
        >>> print(M.apply_second(types.INT64.PLUS, 1))
              0  1  2
          0|   -41   |  0
          1|        1|  1
          2|150      |  2
              0  1  2

        """
        if out is None:
            out = self.__class__.sparse(self.type, self.nrows, self.ncols)
        op = op.get_binaryop(self.type)
        mask, accum, desc = self._get_args(mask, accum, desc)
        if isinstance(second, Scalar):
            f = lib.GxB_Matrix_apply_BinaryOp2nd
            second = second._scalar[0]
        else:
            f = self.type._Matrix_apply_BinaryOp2nd
        self._check(f(out._matrix[0], mask, accum, op, self._matrix[0], second, desc))
        return out

    def select(self, op, thunk=None, out=None, mask=None, accum=None, desc=Default):
        """Select elements that match the given select operation condition.
        Can be a string mapping to following operators:

        Operator | Library Operation | Definition
        ---   | --- | ---
        `>`   | lib.GxB_GT_THUNK | Select greater than 'thunk'.
        `<`   | lib.GxB_LT_THUNK | Select less than 'thunk'.
        `>=`  | lib.GxB_GE_THUNK | Select greater than or equal to 'thunk'.
        `<=`  | lib.GxB_LE_THUNK | Select less than or equal to 'thunk'.
        `!=`  | lib.GxB_NE_THUNK | Select not equal to 'thunk'.
        `==`  | lib.GxB_EQ_THUNK | Select equal to 'thunk'.
        `>0`  | lib.GxB_GT_ZERO  | Select greater than zero.
        `<0`  | lib.GxB_LT_ZERO  | Select less than zero.
        `>=0` | lib.GxB_GE_ZERO  | Select greater than or equal to zero.
        `<=0` | lib.GxB_LE_ZERO  | Select less than or equal to zero.
        `!=0` | lib.GxB_NONZERO  | Select nonzero value.
        `==0` | lib.GxB_EQ_ZERO  | Select equal to zero.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [-42, 0, 149])
        >>> print(M.select('>', 0))
              0  1  2
          0|         |  0
          1|         |  1
          2|149      |  2
              0  1  2

        >>> print(M.select('>=', 0))
              0  1  2
          0|         |  0
          1|        0|  1
          2|149      |  2
              0  1  2

        >>> print(M.select('<', 0))
              0  1  2
          0|   -42   |  0
          1|         |  1
          2|         |  2
              0  1  2

        >>> N = M.dup(clear=True)
        >>> M.select('<', 0, out=N) is N
        True
        >>> print(N)
              0  1  2
          0|   -42   |  0
          1|         |  1
          2|         |  2
              0  1  2

        """
        if out is None:
            out = self.__class__.sparse(self.type, self.nrows, self.ncols)
        if isinstance(op, str):
            op = _get_select_op(op)

        if thunk is None:
            thunk = NULL
        if isinstance(thunk, (bool, int, float, complex)):
            thunk = Scalar.from_value(thunk)
        if isinstance(thunk, Scalar):
            self._keep_alives[self._matrix] = thunk
            thunk = thunk._scalar[0]

        mask, accum, desc = self._get_args(mask, accum, desc)

        self._check(
            lib.GxB_Matrix_select(
                out._matrix[0], mask, accum, op, self._matrix[0], thunk, desc
            )
        )
        return out

    def tril(self, offset=None):
        """Select the lower triangular Matrix.

        The diagonal `offset` can be used to select all below any
        diagonal rank, positive towars the upper right coner and
        negative toward the lower left.

        >>> M = Matrix.dense(types.UINT8, 3, 3)
        >>> print(M.tril())
              0  1  2
          0|  0      |  0
          1|  0  0   |  1
          2|  0  0  0|  2
              0  1  2
        >>> print(M.tril(1))
              0  1  2
          0|  0  0   |  0
          1|  0  0  0|  1
          2|  0  0  0|  2
              0  1  2
        >>> print(M.tril(-1))
              0  1  2
          0|         |  0
          1|  0      |  1
          2|  0  0   |  2
              0  1  2

        """
        return self.select(lib.GxB_TRIL, thunk=offset)

    def triu(self, offset=None):
        """Select the upper triangular Matrix.

        The diagonal `offset` can be used to select all above any
        diagonal rank, positive towars the upper right coner and
        negative toward the lower left.

        >>> M = Matrix.dense(types.UINT8, 3, 3)
        >>> print(M.triu())
              0  1  2
          0|  0  0  0|  0
          1|     0  0|  1
          2|        0|  2
              0  1  2
        >>> print(M.triu(1))
              0  1  2
          0|     0  0|  0
          1|        0|  1
          2|         |  2
              0  1  2
        >>> print(M.triu(-1))
              0  1  2
          0|  0  0  0|  0
          1|  0  0  0|  1
          2|     0  0|  2
              0  1  2

        """
        return self.select(lib.GxB_TRIU, thunk=offset)

    def diag(self, offset=None):
        """Select the diagonal Matrix.

        The diagonal `offset` can be used to select any diagonal rank,
        positive towars the upper right coner and negative toward the
        lower left.

        >>> M = Matrix.dense(types.UINT8, 3, 3)
        >>> print(M.diag())
              0  1  2
          0|  0      |  0
          1|     0   |  1
          2|        0|  2
              0  1  2
        >>> print(M.diag(1))
              0  1  2
          0|     0   |  0
          1|        0|  1
          2|         |  2
              0  1  2
        >>> print(M.diag(-1))
              0  1  2
          0|         |  0
          1|  0      |  1
          2|     0   |  2
              0  1  2

        """
        return self.select(lib.GxB_DIAG, thunk=offset)

    def offdiag(self, offset=None):
        """Select the off-diagonal Matrix.

        The diagonal `offset` can be used to select off any diagonal
        rank, positive towars the upper right coner and negative
        toward the lower left.

        >>> M = Matrix.dense(types.UINT8, 3, 3)
        >>> print(M.offdiag())
              0  1  2
          0|     0  0|  0
          1|  0     0|  1
          2|  0  0   |  2
              0  1  2
        >>> print(M.offdiag(1))
              0  1  2
          0|  0     0|  0
          1|  0  0   |  1
          2|  0  0  0|  2
              0  1  2
        >>> print(M.offdiag(-1))
              0  1  2
          0|  0  0  0|  0
          1|     0  0|  1
          2|  0     0|  2
              0  1  2

        """
        return self.select(lib.GxB_OFFDIAG, thunk=offset)

    def nonzero(self):
        """Select the non-zero Matrix.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 0, 149])
        >>> print(M.nonzero())
              0  1  2
          0|    42   |  0
          1|         |  1
          2|149      |  2
              0  1  2

        """
        return self.select(lib.GxB_NONZERO)

    def _full(self):
        """"""
        B = self.__class__.sparse(self.type, self.nrows, self.ncols)

        self._check(
            self.type._Matrix_assignScalar(
                B._matrix[0],
                NULL,
                NULL,
                self.type.one,
                lib.GrB_ALL,
                0,
                lib.GrB_ALL,
                0,
                NULL,
            )
        )
        return self.eadd(B, self.type.FIRST)

    def _compare(self, other, op, strop):
        C = self.__class__.sparse(types.BOOL, self.nrows, self.ncols)
        if isinstance(other, (bool, int, float, complex)):
            if op(other, 0):
                B = self.__class__.dup(self)
                B[:, :] = other
                self.emult(B, strop, out=C)
                return C
            else:
                self.select(strop, other).apply(types.BOOL.ONE, out=C)
                return C
        elif isinstance(other, Matrix):
            A = self._full()
            B = other._full()
            A.emult(B, strop, out=C)
            return C
        else:
            raise TypeError("Unknown matrix comparison type.")

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

    def _get_args(self, mask=None, accum=None, desc=Default):
        if isinstance(mask, Matrix):
            mask = mask._matrix[0]
        elif isinstance(mask, Vector):
            mask = mask._vector[0]
        else:
            mask = NULL
        if accum is None:
            accum = current_accum.get(NULL)
        if accum is not NULL:
            accum = accum.get_binaryop(self.type)
        if desc is None or desc == Default:
            desc = current_desc.get(Default)

        desc = desc.desc[0]
        return mask, accum, desc

    def mxm(
        self,
        other,
        cast=None,
        out=None,
        semiring=None,
        mask=None,
        accum=None,
        desc=Default,
    ):
        """Matrix-matrix multiply.

        Multiply this matrix by `other` matrix.

        See Section 9.6 in the [SuiteSparse User
        Guide](https://raw.githubusercontent.com/DrTimothyAldenDavis/GraphBLAS/stable/Doc/GraphBLAS_UserGuide.pdf)
        for details.

        `mxm` can be called directly or with the `@` operator:

        >>> m = Matrix.from_lists([0, 1, 2], [1, 2, 0], [1, 2, 3])
        >>> n = Matrix.from_lists([0, 1, 2], [1, 2, 0], [2, 3, 4])
        >>> print(m)
              0  1  2
          0|     1   |  0
          1|        2|  1
          2|  3      |  2
              0  1  2

        >>> print(n)
              0  1  2
          0|     2   |  0
          1|        3|  1
          2|  4      |  2
              0  1  2

        >>> o = m.mxm(n)
        >>> print(o)
              0  1  2
          0|        3|  0
          1|  8      |  1
          2|     6   |  2
              0  1  2

        >>> o = m @ n
        >>> print(o)
              0  1  2
          0|        3|  0
          1|  8      |  1
          2|     6   |  2
              0  1  2

        >>> p = m.pattern() @ n
        >>> print(o)
              0  1  2
          0|        3|  0
          1|  8      |  1
          2|     6   |  2
              0  1  2

        >>> p = m @ n.pattern()
        >>> print(o)
              0  1  2
          0|        3|  0
          1|  8      |  1
          2|     6   |  2
              0  1  2

        By default, `mxm` and `@` create a new result matrix of the
        correct type and dimensions if one is not provided.  If you
        want to provide your own matrix to put the result in, you can
        pass it in the `out` parameter.  This is useful for
        accumulating results into a single matrix with minimal
        copying.  This is also supported by the `@=` syntax:

        >>> o = m.dup()
        >>> o.mxm(n, accum=types.INT64.MIN, out=o) is o
        True
        >>> print(o)
              0  1  2
          0|     1  3|  0
          1|  8     2|  1
          2|  3  6   |  2
              0  1  2

        >>> o = m.dup()
        >>> with Accum(types.INT64.MIN):
        ...     o @= n
        >>> print(o)
              0  1  2
          0|     1  3|  0
          1|  8     2|  1
          2|  3  6   |  2
              0  1  2

        The default semiring depends on the infered result type.  In
        the case of numbers, the default semiring is `PLUS_TIMES`.  In
        the case of type `BOOL`, it is `BOOL.LOR_LAND`.

        An explicit semiring can be passed to the method or provided
        with a context manager:

        >>> o = m.mxm(n, semiring=types.INT64.MIN_PLUS)
        >>> print(o)
              0  1  2
          0|        4|  0
          1|  6      |  1
          2|     5   |  2
              0  1  2

        >>> with types.INT64.MIN_PLUS:
        ...     o = m @ n
        >>> print(o)
              0  1  2
          0|        4|  0
          1|  6      |  1
          2|     5   |  2
              0  1  2

        Descriptors and accumulators can also be provided as an
        argument or a context manager:

        >>> descriptor.T0
        <Descriptor T0>

        >>> o = m.mxm(n, desc=descriptor.T0)
        >>> print(o)
              0  1  2
          0| 12      |  0
          1|     2   |  1
          2|        6|  2
              0  1  2

        >>> with descriptor.T0:
        ...     o = m @ n
        >>> print(o)
              0  1  2
          0| 12      |  0
          1|     2   |  1
          2|        6|  2
              0  1  2

        The accumulator context manager requires an extra `Accum`
        helper class to distinguish it from binary ops used in `eadd`
        and `emult`.

        """
        if semiring is None:
            semiring = current_semiring.get(NULL)

        if out is None:
            if semiring is not NULL:
                typ = semiring.ztype
            else:
                typ = cast or types.promote(self.type, other.type)
            out = self.__class__.sparse(typ, self.nrows, other.ncols)
        else:
            typ = out.type

        if semiring is NULL:
            semiring = out.type._default_semiring()

        semiring = semiring.get_semiring()
        mask, accum, desc = self._get_args(mask, accum, desc)
        self._check(
            lib.GrB_mxm(
                out._matrix[0],
                mask,
                accum,
                semiring,
                self._matrix[0],
                other._matrix[0],
                desc,
            )
        )
        return out

    def mxv(
        self,
        other,
        cast=None,
        out=None,
        semiring=None,
        mask=None,
        accum=None,
        desc=Default,
    ):
        """Matrix-vector multiply.

        Multiply this matrix by `other` column vector "on the right".
        For row vector multiplication "on the left" see `Vector.vxm`.

        See Section 9.6 in the [SuiteSparse User
        Guide](https://raw.githubusercontent.com/DrTimothyAldenDavis/GraphBLAS/stable/Doc/GraphBLAS_UserGuide.pdf)
        for details.

        `mxv` can also be called directly or with the `@` operator:

        >>> m = Matrix.from_lists([0, 1, 2], [1, 2, 0], [1, 2, 3])
        >>> v = Vector.from_lists([0, 1, 2], [2, 3, 4])
        >>> o = m.mxv(v)
        >>> print(o)
        0| 3
        1| 8
        2| 6
        >>> o = m @ v
        >>> print(o)
        0| 3
        1| 8
        2| 6

        By default, `mxv` and `@` create a new result matrix of the
        correct type and dimensions if one is not provided.  If you
        want to provide your own matrix to put the result in, you can
        pass it in the `out` parameter.  This is useful for
        accumulating results into a single matrix with minimal
        copying.

        >>> o = v.dup()
        >>> m.mxv(v, accum=types.INT64.PLUS, out=o) is o
        True
        >>> print(o)
        0| 5
        1|11
        2|10

        The default semiring depends on the infered result type.  In
        the case of numbers, the default semiring is `PLUS_TIMES`.  In
        the case of type `BOOL`, it is `BOOL.LOR_LAND`.

        An explicit semiring can be passed to the method or provided
        with a context manager:

        >>> o = m.mxv(v, semiring=types.INT64.MIN_PLUS)
        >>> print(o)
        0| 4
        1| 6
        2| 5

        >>> with types.INT64.MIN_PLUS:
        ...     o = m @ v
        >>> print(o)
        0| 4
        1| 6
        2| 5

        Descriptors and accumulators can also be provided as an
        argument or a context manager:

        >>> o = m.mxv(v, desc=descriptor.T0)
        >>> print(o)
        0|12
        1| 2
        2| 6

        >>> with descriptor.T0:
        ...     o = m @ v
        >>> print(o)
        0|12
        1| 2
        2| 6

        >>> del o[1]
        >>> o = m.mxv(v, mask=o)
        >>> print(o)
        0| 3
        1|
        2| 6

        """

        if semiring is None:
            semiring = current_semiring.get(NULL)

        if out is None:
            new_dimension = self.ncols if T0 in desc else self.nrows
            if semiring is not NULL:
                typ = semiring.ztype
            else:
                typ = cast or types.promote(self.type, other.type)
            out = Vector.sparse(typ, new_dimension)
        else:
            typ = out.type

        if semiring is NULL:
            semiring = out.type._default_semiring()

        semiring = semiring.get_semiring()
        mask, accum, desc = self._get_args(mask, accum, desc)

        self._check(
            lib.GrB_mxv(
                out._vector[0],
                mask,
                accum,
                semiring,
                self._matrix[0],
                other._vector[0],
                desc,
            )
        )
        return out

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            return self.mxm(other)
        elif isinstance(other, Vector):
            return self.mxv(other)
        else:
            raise TypeError("Right argument to @ must be Matrix or Vector.")

    def __imatmul__(self, other):
        return self.mxm(other, out=self)

    def kronecker(
        self, other, op=None, cast=None, out=None, mask=None, accum=None, desc=Default
    ):
        """[Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product).

        >>> n = Matrix.from_lists([0, 1, 2], [1, 2, 0], [2, 3, 4])
        >>> m = Matrix.dense(types.UINT64, 3, 3, fill=1)
        >>> print(n.kronecker(m))
              0  1  2  3  4  5  6  7  8
          0|           2  2  2         |  0
          1|           2  2  2         |  1
          2|           2  2  2         |  2
          3|                    3  3  3|  3
          4|                    3  3  3|  4
          5|                    3  3  3|  5
          6|  4  4  4                  |  6
          7|  4  4  4                  |  7
          8|  4  4  4                  |  8
              0  1  2  3  4  5  6  7  8

        >>> o = Matrix.sparse(types.UINT64, 9, 9)
        >>> m.kronecker(n, out=o) is o
        True
        >>> print(o)
              0  1  2  3  4  5  6  7  8
          0|     2        2        2   |  0
          1|        3        3        3|  1
          2|  4        4        4      |  2
          3|     2        2        2   |  3
          4|        3        3        3|  4
          5|  4        4        4      |  5
          6|     2        2        2   |  6
          7|        3        3        3|  7
          8|  4        4        4      |  8
              0  1  2  3  4  5  6  7  8

        >>>
        >>> print(m.kronecker(n, op=types.UINT64.MIN))
              0  1  2  3  4  5  6  7  8
          0|     1        1        1   |  0
          1|        1        1        1|  1
          2|  1        1        1      |  2
          3|     1        1        1   |  3
          4|        1        1        1|  4
          5|  1        1        1      |  5
          6|     1        1        1   |  6
          7|        1        1        1|  7
          8|  1        1        1      |  8
              0  1  2  3  4  5  6  7  8
        """
        mask, accum, desc = self._get_args(mask, accum, desc)
        typ = cast or types.promote(self.type, other.type)
        if out is None:
            out = self.__class__.sparse(
                typ, self.nrows * other.nrows, self.ncols * other.ncols
            )
        if op is None:
            op = current_binop.get(self.type.TIMES)

        op = op.get_binaryop(self.type, other.type)

        self._check(
            lib.GrB_Matrix_kronecker_BinaryOp(
                out._matrix[0], mask, accum, op, self._matrix[0], other._matrix[0], desc
            )
        )
        return out

    def extract_matrix(
        self,
        row_index=None,
        col_index=None,
        out=None,
        mask=None,
        accum=None,
        desc=Default,
    ):
        """Extract a submatrix.

        `GrB_Matrix_extract` extracts a submatrix from another matrix.
        The input matrix may be transposed first, via the descriptor.
        The result type remains the same.

        `row_index` and `col_index` can be slice objects that default
        to the equivalent of GrB_ALL.  Python slice objects support
        the SuiteSparse extensions for `GxB_RANGE`, `GxB_BACKWARDS`
        and `GxB_STRIDE`.  See the User Guide for details.

        The size of `C` is `|row_index|`-by-`|col_index|`.  Entries
        outside that sub-range are not accessed and do not take part
        in the computation.


        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 0, 149])
        >>> print(M.extract_matrix())
              0  1  2
          0|    42   |  0
          1|        0|  1
          2|149      |  2
              0  1  2

        >>> print(M.extract_matrix(0, 1))
              0
          0| 42|  0
              0

        >>> O = Matrix.sparse(types.UINT64, 1, 1)
        >>> M.extract_matrix(0, 1, out=O) is O
        True
        >>> print(O)
              0
          0| 42|  0
              0

        >>> print(M.extract_matrix(slice(1,2), 2))
              0
          0|  0|  0
          1|   |  1
              0

        >>> print(M.extract_matrix(0, slice(0,1)))
              0  1
          0|    42|  0
              0  1

        >>> N = Matrix.from_lists([1, 2], [2, 0], [True, True])
        >>> print(M[N])
              0  1  2
          0|         |  0
          1|        0|  1
          2|149      |  2
              0  1  2

        """
        ta = T0 in desc
        mask, accum, desc = self._get_args(mask, accum, desc)
        result_nrows = self.ncols if ta else self.nrows
        result_ncols = self.nrows if ta else self.ncols
        if isinstance(row_index, int):
            I, ni, isize = _build_range(slice(row_index, row_index), result_nrows - 1)
        else:
            I, ni, isize = _build_range(row_index, result_nrows - 1)
        if isinstance(col_index, int):
            J, nj, jsize = _build_range(slice(col_index, col_index), result_ncols - 1)
        else:
            J, nj, jsize = _build_range(col_index, result_ncols - 1)

        if isize is None:
            isize = result_nrows
        if jsize is None:
            jsize = result_ncols

        if out is None:
            out = self.__class__.sparse(self.type, isize, jsize)

        self._check(
            lib.GrB_Matrix_extract(
                out._matrix[0], mask, accum, self._matrix[0], I, ni, J, nj, desc
            )
        )
        return out

    def extract_col(
        self, col_index, row_slice=None, out=None, mask=None, accum=None, desc=Default
    ):
        """Extract a column Vector.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 0, 149])
        >>> print(M)
              0  1  2
          0|    42   |  0
          1|        0|  1
          2|149      |  2
              0  1  2
        >>> print(M.extract_col(0))
        0|
        1|
        2|149

        >>> v = Vector.sparse(types.UINT64, M.ncols)
        >>> M.extract_col(0, out=v) is v
        True
        >>> print(v)
        0|
        1|
        2|149

        """
        stop_val = self.ncols if T0 in desc else self.nrows
        if out is None:
            out = Vector.sparse(self.type, stop_val)

        mask, accum, desc = self._get_args(mask, accum, desc)
        I, ni, size = _build_range(row_slice, stop_val)

        self._check(
            lib.GrB_Col_extract(
                out._vector[0], mask, accum, self._matrix[0], I, ni, col_index, desc
            )
        )
        return out

    def extract_row(
        self, row_index, col_slice=None, out=None, mask=None, accum=None, desc=Default
    ):
        """Extract a row Vector.


        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 0, 149])
        >>> print(M)
              0  1  2
          0|    42   |  0
          1|        0|  1
          2|149      |  2
              0  1  2
        >>> print(M.extract_row(0))
        0|
        1|42
        2|

        """
        desc = desc & T0
        return self.extract_col(
            row_index, col_slice, out, desc=desc, mask=None, accum=None
        )

    def __getitem__(self, index):
        if isinstance(index, int):
            # a[3] extract single row
            return self.extract_row(index, None)
        if isinstance(index, slice):
            # a[3:] extract submatrix of rows
            return self.extract_matrix(index, None)

        if isinstance(index, Matrix):
            return self.extract_matrix(mask=index)

        if not isinstance(index, (tuple, list)):
            raise TypeError

        i0 = index[0]
        i1 = index[1]
        if isinstance(i0, int) and isinstance(i1, int):
            # a[3,3] extract single element
            result = self.type.ffi.new(self.type._ptr)
            self._check(
                self.type._Matrix_extractElement(
                    result, self._matrix[0], index[0], index[1]
                )
            )
            return self.type.to_value(result[0])

        if isinstance(i0, int) and isinstance(i1, slice):
            # a[3,:] extract slice of row vector
            return self.extract_row(i0, i1)

        if isinstance(i0, slice) and isinstance(i1, int):
            # a[:,3] extract slice of col vector
            return self.extract_col(i1, i0)

        # a[:,:] or a[[0,1,2], [3,4,5]] extract submatrix with slice or row/col indices
        return self.extract_matrix(i0, i1)

    def assign_col(
        self, col_index, value, row_slice=None, mask=None, accum=None, desc=Default
    ):
        """Assign a vector to a column.

        >>> M = Matrix.sparse(types.BOOL, 3, 3)
        >>> M.assign_col(1, Vector.from_lists([1, 2], [True, True], 3))
        >>> print(M)
              0  1  2
          0|         |  0
          1|     t   |  1
          2|     t   |  2
              0  1  2

        """
        stop_val = self.ncols if T0 in desc else self.nrows
        I, ni, size = _build_range(row_slice, stop_val)
        mask, accum, desc = self._get_args(mask, accum, desc)

        self._check(
            lib.GrB_Col_assign(
                self._matrix[0], mask, accum, value._vector[0], I, ni, col_index, desc
            )
        )

    def assign_row(
        self, row_index, value, col_slice=None, mask=None, accum=None, desc=Default
    ):
        """Assign a vector to a row.

        >>> M = Matrix.sparse(types.BOOL, 3, 3)
        >>> M.assign_row(1, Vector.from_lists([1, 2], [True, True], 3))
        >>> print(M)
              0  1  2
          0|         |  0
          1|     t  t|  1
          2|         |  2
              0  1  2

        """
        stop_val = self.nrows if T0 in desc else self.ncols
        I, ni, size = _build_range(col_slice, stop_val)

        mask, accum, desc = self._get_args(mask, accum, desc)
        self._check(
            lib.GrB_Row_assign(
                self._matrix[0], mask, accum, value._vector[0], row_index, I, ni, desc
            )
        )

    def assign_matrix(
        self, value, rindex=None, cindex=None, mask=None, accum=None, desc=Default
    ):
        """Assign a submatrix.

        >>> M = Matrix.sparse(types.BOOL, 3, 3)
        >>> S = Matrix.sparse(types.BOOL, 3, 3)
        >>> S[1,1] = True
        >>> M.assign_matrix(S)
        >>> print(M)
              0  1  2
          0|         |  0
          1|     t   |  1
          2|         |  2
              0  1  2

        >>> M.clear()

        Masked assignment with `M[key] = value` syntax can be done
        with if the index and value arguments are type `Matrix`:

        >>> M[S] = S
        >>> print(M)
              0  1  2
          0|         |  0
          1|     t   |  1
          2|         |  2
              0  1  2

        """
        I, ni, isize = _build_range(rindex, self.nrows - 1)
        J, nj, jsize = _build_range(cindex, self.ncols - 1)
        isize = self.nrows
        jsize = self.ncols

        mask, accum, desc = self._get_args(mask, accum, desc)

        self._check(
            lib.GrB_Matrix_assign(
                self._matrix[0], mask, accum, value._matrix[0], I, ni, J, nj, desc
            )
        )

    def assign_scalar(
        self, value, row_slice=None, col_slice=None, mask=None, accum=None, desc=Default
    ):
        """Assign a scalar `value` to the Matrix.

        >>> M = Matrix.sparse(types.BOOL, 3, 3)

        The values of `row_slice` and `col_slice` determine what
        elements are assigned to the Matrix.  The value `None` maps to
        the GraphBLAS symbol `lib.GrB_ALL`, so the default behavior,
        with no other arguments, assigns the scalar to all elements:

        >>> M.assign_scalar(True)
        >>> print(M)
              0  1  2
          0|  t  t  t|  0
          1|  t  t  t|  1
          2|  t  t  t|  2
              0  1  2
        >>> M.clear()

        This is the same as the slice syntax with a bare colon:

        >>> M[:,:] = True
        >>> print(M)
              0  1  2
          0|  t  t  t|  0
          1|  t  t  t|  1
          2|  t  t  t|  2
              0  1  2
        >>> M.clear()

        If `row_slice` or `col_slice` is an integer, use it as an
        index to one row or column:

        >>> M.assign_scalar(True, 1)
        >>> print(M)
              0  1  2
          0|         |  0
          1|  t  t  t|  1
          2|         |  2
              0  1  2
        >>> M.clear()

        An integer index and a scalar does row assignment:

        >>> M[1] = True
        >>> print(M)
              0  1  2
          0|         |  0
          1|  t  t  t|  1
          2|         |  2
              0  1  2
        >>> M.clear()

        this is the same as the syntax:

        >>> M[1,:] = True
        >>> print(M)
              0  1  2
          0|         |  0
          1|  t  t  t|  1
          2|         |  2
              0  1  2
        >>> M.clear()

        If `col_slice` is an integer, it does column assignment:

        >>> M.assign_scalar(True, None, 1)
        >>> print(M)
              0  1  2
          0|     t   |  0
          1|     t   |  1
          2|     t   |  2
              0  1  2
        >>> M.clear()

        Which is the same as the syntax:

        >>> M[:,1] = True
        >>> print(M)
              0  1  2
          0|     t   |  0
          1|     t   |  1
          2|     t   |  2
              0  1  2
        >>> M.clear()

        Just an integer index does a row assignment:

        >>> M.clear()
        >>> M[1] = Vector.from_lists([0,1], [True, True],3)
        >>> print(M)
              0  1  2
          0|         |  0
          1|  t  t   |  1
          2|         |  2
              0  1  2
        >>> M.clear()

        >>> M[0:1,0:1] = True
        >>> print(M)
              0  1  2
          0|  t  t   |  0
          1|  t  t   |  1
          2|         |  2
              0  1  2
        >>> M.clear()

        """
        mask, accum, desc = self._get_args(mask, accum, desc)
        if row_slice is not None:
            if isinstance(row_slice, int):
                I, ni, isize = _build_range(slice(row_slice, row_slice), self.nrows - 1)
            else:
                I, ni, isize = _build_range(row_slice, self.nrows - 1)
        else:
            I = lib.GrB_ALL
            ni = 0
        if col_slice is not None:
            if isinstance(col_slice, int):
                J, nj, jsize = _build_range(slice(col_slice, col_slice), self.ncols - 1)
            else:
                J, nj, jsize = _build_range(col_slice, self.ncols - 1)
        else:
            J = lib.GrB_ALL
            nj = 0
        scalar_type = types._gb_from_type(type(value))
        self._check(
            scalar_type._Matrix_assignScalar(
                self._matrix[0], mask, accum, value, I, ni, J, nj, desc
            )
        )

    def __setitem__(self, index, value):
        if isinstance(index, int):
            # A[3]
            if isinstance(value, Vector):
                # A[3] = Vector
                return self.assign_row(index, value)
            if isinstance(value, (bool, int, float, complex)):
                # A[3] = scalar
                return self.assign_scalar(value, index)
            raise TypeError

        elif isinstance(index, slice):
            if isinstance(value, Matrix):
                # A[3:] = assign submatrix to rows
                self.assign_matrix(value, index, None)
                return
            if isinstance(value, (bool, int, float, complex)):
                # A[3:] = 3 assign scalar to rows
                self.assign_scalar(value, index, None)
                return
            raise TypeError

        elif isinstance(index, Matrix):
            if isinstance(value, Matrix):
                # A[M] = B masked matrix assignment
                self.assign_matrix(value, mask=index)
                return
            if not isinstance(value, (bool, int, float, complex)):
                raise TypeError
            # A[M] = s masked scalar assignment
            self.assign_scalar(value, mask=index)
            return

        elif not isinstance(index, (tuple, list)):
            raise TypeError

        i0 = index[0]
        i1 = index[1]
        if isinstance(i0, int) and isinstance(i1, int):
            val = self.type.from_value(value)
            self._check(self.type._Matrix_setElement(self._matrix[0], val, i0, i1))
            return

        if isinstance(i0, int) and isinstance(i1, slice):
            # a[3,:] assign slice of row vector or scalar
            if isinstance(value, Vector):
                self.assign_row(i0, value, i1)
            else:
                self.assign_scalar(value, i0, i1)
            return

        if isinstance(i0, slice) and isinstance(i1, int):
            # a[:,3] extract slice of col vector or scalar
            if isinstance(value, Vector):
                self.assign_col(i1, value, i0)
            else:
                self.assign_scalar(value, i0, i1)
            return

        if isinstance(i0, slice) and isinstance(i1, slice):
            if isinstance(value, (bool, int, float, complex)):
                self.assign_scalar(value, i0, i1)
                return
            else:
                # a[:,:] assign submatrix
                self.assign_matrix(value, i0, i1)
                return
        raise TypeError

    def __delitem__(self, index):
        if (
            not isinstance(index, tuple)
            or not isinstance(index[0], int)
            or not isinstance(index[1], int)
        ):
            raise TypeError(
                "__delitem__ currently only supports single element removal"
            )
        self._check(lib.GrB_Matrix_removeElement(self._matrix[0], index[0], index[1]))

    def __contains__(self, index):
        try:
            v = self[index]
            return True
        except NoValue:
            return False

    def get(self, i, j, default=None):
        """Get the element at row `i` col `j` or return the default value if
        the element is not present.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 0, 149])
        >>> M.get(1, 2)
        0
        >>> M.get(0, 0) is None
        True
        >>> M.get(0, 0, 'foo')
        'foo'

        """
        try:
            return self[i, j]
        except NoValue:
            return default

    def wait(self):
        """Wait for this Matrix to complete before allowing another thread to
        change it.

        """
        self._check(lib.GrB_Matrix_wait(self._matrix))

    def to_markdown_table(self, title="A", width=2):
        """Return a string markdown table representation of the Matrix.

        >>> M = Matrix.from_lists([0, 0, 1, 2], [1, 2, 2, 0], [42, 2, 0, 149])
        >>> print(M.to_markdown_table())
        A|0|1|2
        ---|---|---|---
        0|   |42| 2
        1|   |  | 0
        2| 149|  |

        """
        rows = set(self.rows)
        cols = set(self.cols)
        result = f"""\
{title}|{'|'.join(map(str, cols))}
---|{"|".join(['---'] * len(cols))}
"""
        for i, row in enumerate(rows):
            result += f"{row}| " + "|".join(
                self.type.format_value(self.get(row, col, ""), width) for col in cols
            )
            if i != len(rows) - 1:
                result += "\n"
        return result.rstrip()

    def to_html_table(self, title="A", width=2):
        """Return a string markdown table representation of the Matrix.

        >>> M = Matrix.from_lists([0, 0, 1, 2], [1, 2, 2, 0], [42, 2, 0, 149])
        >>> print(M.to_html_table())
                <table>
                    <th>A</th>
                        <th>0</th>
                        <th>1</th>
                        <th>2</th>
        <BLANKLINE>
                    <tr>
                    <th>0</th>
                        <td>  </td>
                        <td>42</td>
                        <td> 2</td>
                    </tr>
        <BLANKLINE>
                    <tr>
                    <th>1</th>
                        <td>  </td>
                        <td>  </td>
                        <td> 0</td>
                    </tr>
        <BLANKLINE>
                    <tr>
                    <th>2</th>
                        <td>149</td>
                        <td>  </td>
                        <td>  </td>
                    </tr>
                </table>
        """
        from mako.template import Template

        t = Template(
            """\
        <%
            rows = set(A.rows)
            cols = set(A.cols)
        %><table>
            <th>${title}</th>
            % for col in cols:
                <th>${col}</th>
            % endfor
            % for row in rows:
                ${makerow(row)}
            % endfor
        </table><%def name="makerow(row)">
            <tr>
            <th>${row}</th>
            % for col in cols:
                <td>${A.type.format_value(A.get(row, col, ''))}</td>
            % endfor
            </tr></%def>"""
        )
        return t.render(A=self, title=title)

    def to_string(
        self, format_string="{:>%s}", width=3, prec=5, empty_char="", cell_sep=""
    ):
        """Return a string representation of the Matrix.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 0, 149])
        >>> M.to_string()
        '      0  1  2\\n  0|    42   |  0\\n  1|        0|  1\\n  2|149      |  2\\n      0  1  2'
        """
        format_string = format_string % width
        header = (
            format_string.format("")
            + " "
            + "".join(format_string.format(i) for i in range(self.ncols))
        )
        result = header + "\n"
        for row in range(self.nrows):
            result += format_string.format(row) + "|"
            for col in range(self.ncols):
                value = self.get(row, col, empty_char)
                result += cell_sep + self.type.format_value(value, width, prec)
            result += "|  " + str(row) + "\n"
        result += header

        return result

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return "<Matrix (%sx%s : %s:%s)>" % (
            self.nrows,
            self.ncols,
            self.nvals,
            self.type.__name__,
        )

    def to_scipy_sparse(self, format="csr"):
        """Return a scipy sparse matrix of this Matrix.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 0, 149])
        >>> M.to_scipy_sparse()
        <3x3 sparse matrix of type '<class 'numpy.int64'>'...

        """
        from scipy import sparse

        rows, cols, vals = self.to_arrays()
        s = sparse.coo_matrix((vals, (rows, cols)), shape=self.shape)
        if format == "coo":
            return s
        if format not in {"bsr", "csr", "csc", "coo", "lil", "dia", "dok"}:
            raise TypeError(f"Invalid format: {format}")
        return s.asformat(format)

    def to_numpy(self):
        """Return a dense numpy matrix of this Matrix.

        >>> M = Matrix.from_lists([0, 1, 2], [1, 2, 0], [42, 0, 149])
        >>> M.to_numpy()
        array([[  0,  42,   0],
               [  0,   0,   0],
               [149,   0,   0]], dtype=int64)
        """
        s = self.to_scipy_sparse("coo")
        return s.toarray()
