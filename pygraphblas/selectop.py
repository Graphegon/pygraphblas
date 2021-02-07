"""Contains all automatically generated SelectOps from CFFI.

This module contains the built-in select ops from SuiteSparse.  Unlike
other operators they can be type generic so they are kept here:

>>> from pygraphblas import Matrix, selectop
>>> A = Matrix.from_lists([0, 0, 1], [0, 1, 1], [-1, 0, 1])
>>> print(A.select(selectop.LT_THUNK, 0))
      0  1
  0| -1   |  0
  1|      |  1
      0  1

Note that all of these operators are already wrapped as string names
by `Matrix.select()` and variants like `Matrix.tril()`.  You should
use those.

"""

__all__ = ["SelectOp", "select_op"]

import sys
from textwrap import dedent
import numba
from numba import cfunc, jit

from .base import lib, ffi as core_ffi, _check
from . import types


class SelectOp:
    """Wrapper around GrB_SelectOpl"""

    __slots__ = ("name", "selectop")

    def __init__(self, name, op):
        self.name = name
        self.selectop = op

    def get_selectop(self):
        return self.selectop

    def print(self, level=2, name="", f=sys.stdout):  # pragma: nocover
        """Print the matrix using `GxB_Matrix_fprint()`, by default to
        `sys.stdout`.

        Level 1: Short description
        Level 2: Short list, short numbers
        Level 3: Long list, short number
        Level 4: Short list, long numbers
        Level 5: Long list, long numbers

        """
        _check(lib.GxB_SelectOp_fprint(self.selectop, bytes(name, "utf8"), level, f))


_lib_ops = [
    "GxB_TRIL",
    "GxB_TRIU",
    "GxB_DIAG",
    "GxB_OFFDIAG",
    "GxB_NONZERO",
    "GxB_EQ_ZERO",
    "GxB_GT_ZERO",
    "GxB_GE_ZERO",
    "GxB_LT_ZERO",
    "GxB_LE_ZERO",
    "GxB_NE_THUNK",
    "GxB_EQ_THUNK",
    "GxB_GT_THUNK",
    "GxB_GE_THUNK",
    "GxB_LT_THUNK",
    "GxB_LE_THUNK",
]


def build_selectops(__pdoc__):
    this = sys.modules[__name__]
    for n in _lib_ops:
        lop = getattr(lib, n)
        n = "_".join(n.split("_")[1:])
        sop = SelectOp(n, lop)
        setattr(this, n, sop)
        __all__.append(n)
        __pdoc__[f"selectop.{n}"] = f"SelectOp {n}"


def _uop_name(name):  # pragma: nocover
    return "_{0}_selectop_function".format(name)


def _build_uop_def(name, arg_type, result_type):  # pragma: nocover
    decl = dedent(
        """
    typedef void (*{0})({1}*, {1}*);
    """.format(
            _uop_name(name), arg_type, result_type
        )
    )
    return decl


def select_op(arg_type, thunk_type=None):
    """Decorator to jit-compile Python function into a GrB_BinaryOp
    object.

    >>> from random import random
    >>> from pygraphblas import Matrix, select_op, types, gviz, descriptor
    >>> @select_op(types.FP64, types.FP64)
    ... def random_gt(i, j, x, v):
    ...     if random() > v:
    ...         return True
    ...     return False
    >>> A = Matrix.dense(types.FP64, 3, 3, fill=1)
    >>> A.select(random_gt, 0.5, out=A, desc=descriptor.R) is A
    True
    >>> ga = gviz.draw_matrix(A, scale=40,
    ...     filename='/docs/imgs/select_op_A')

    ![select_op_A.png](../imgs/select_op_A.png)

    >>> @select_op(types.FP64)
    ... def coin_flip(i, j, x, v):
    ...     if random() > 0.5:
    ...         return True
    ...     return False
    >>> A = Matrix.dense(types.FP64, 3, 3, fill=1)
    >>> A.select(coin_flip)
    <Matrix (3x3 : ...:FP64)>
    """
    if thunk_type is not None:
        thunk_type = thunk_type._gb_type
    else:
        thunk_type = core_ffi.NULL

    def inner(func):
        func_name = func.__name__
        sig = numba.boolean(
            numba.types.uint64,
            numba.types.uint64,
            numba.types.CPointer(arg_type._numba_t),
            numba.types.CPointer(arg_type._numba_t),
        )
        jitfunc = numba.jit(func, nopython=True)

        @numba.cfunc(sig, nopython=True)
        def wrapper(i, j, x, t):  # pragma: no cover
            return jitfunc(i, j, x[0], t[0])

        out = core_ffi.new("GxB_SelectOp*")
        lib.GxB_SelectOp_new(
            out,
            core_ffi.cast("GxB_select_function", wrapper.address),
            arg_type._gb_type,
            thunk_type,
        )

        return SelectOp(func_name, out[0])

    return inner
