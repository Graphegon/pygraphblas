"""Contains all automatically generated UnaryOps from CFFI.

"""

__all__ = ["UnaryOp", "unary_op"]

import re, sys
from itertools import chain
from textwrap import dedent
from cffi import FFI
import numba
from numba import cfunc, jit, carray
from numba.core.typing import cffi_utils as cffi_support
import contextvars
from collections import defaultdict

from .base import lib, ffi as core_ffi, _check
from . import types


class UnaryOp:
    """Wrapper around GrB_UnaryOpl"""

    __slots__ = ("name", "unaryop", "ffi", "token")

    def __init__(self, name, typ, op):
        self.name = "_".join((name, typ))
        self.unaryop = op
        self.token = None
        cls = getattr(types, typ)
        setattr(cls, name, self)
        types.__pdoc__[f"{typ}.{name}"] = f"UnaryOp {typ}.{name}"

    def get_unaryop(self, operand1=None):
        return self.unaryop

    def print(self, level=2, name="", f=sys.stdout):  # pragma: nocover
        """Print the matrix using `GxB_Matrix_fprint()`, by default to
        `sys.stdout`.

        Level 1: Short description
        Level 2: Short list, short numbers
        Level 3: Long list, short number
        Level 4: Short list, long numbers
        Level 5: Long list, long numbers

        """
        _check(lib.GxB_UnaryOp_fprint(self.unaryop, bytes(name, "utf8"), level, f))


uop_re = re.compile(
    "^(GrB|GxB)_(ONE|ABS|SQRT|LOG|EXP|LOG2|SIN|COS|TAN|ACOS|ASIN|ATAN|SINH|"
    "POSITIONI|POSITIONI1|POSITIONJ|POSITIONJ1|"
    "COSH|TANH|ACOSH|ASINH|ATANH|SIGNUM|CEIL|FLOOR|ROUND|TRUNC|EXP2|EXPM1|"
    "LOG12|LOG1P|LGAMMA|TGAMMA|ERF|ERFC|FREXPX|FREXPE|CONJ|CREAL|CIMAG|CARG|"
    "IDENTITY|AINV|MINV|LNOT|ONE|ABS|ISINF|ISNAN|ISFINITE)_"
    "(BOOL|UINT8|UINT16|UINT32|UINT64|INT8|INT16|INT32|INT64|FP32|FP64|FC32|FC64)$"
)


def uop_group(reg):
    srs = []
    for n in filter(None, [reg.match(i) for i in dir(lib)]):
        prefix, op, typ = n.groups()
        srs.append(UnaryOp(op, typ, getattr(lib, n.string)))
    return srs


def build_unaryops(__pdoc__):
    import tempfile

    this = sys.modules[__name__]
    for r in chain(uop_group(uop_re)):
        setattr(this, r.name, r)
        op, typ = r.name.split("_")
        f = tempfile.TemporaryFile()
        r.print(f=f)
        f.seek(0)
        __pdoc__[f"{typ}.{op}"] = f"""```{str(f.read(), 'utf8')}```"""


def _uop_name(name):  # pragma: nocover
    return "_{0}_uop_function".format(name)


def _build_uop_def(name, arg_type, result_type):  # pragma: nocover
    decl = dedent(
        """
    typedef void (*{0})({1}*, {1}*);
    """.format(
            _uop_name(name), arg_type, result_type
        )
    )
    return decl


def unary_op(arg_type, result_type=None, boolean=False):  # pragma: nocover
    """Decorator to jit-compile Python function into a GrB_BinaryOp
    object.

    >>> from random import random
    >>> from pygraphblas import Matrix, binary_op, types, gviz
    >>> @unary_op(types.FP64)
    ... def random(x, y):
    ...     return random(x, y)
    >>> A = Matrix.dense(types.FP64, 3, 3, fill=0)
    >>> A.apply(random)

    >>> ga = gviz.draw_matrix(A, scale=40,
    ...     filename='/docs/imgs/unary_op_A')


    ![![unary_op_B.png](../imgs/unary_op_B.png)

    """
    if result_type is None:
        result_type = arg_type

    def inner(func):
        func_name = func.__name__
        sig = numba.void(
            numba.types.CPointer(numba.boolean)
            if boolean
            else numba.types.CPointer(arg_type.numba_t),
            numba.types.CPointer(arg_type.numba_t),
        )
        jitfunc = jit(func, nopython=True)

        @cfunc(sig, nopython=True)
        def wrapper(z, x):
            result = jitfunc(x[0])
            z[0] = result

        out = core_ffi.new("GrB_UnaryOp*")
        lib.GrB_UnaryOp_new(
            out,
            core_ffi.cast("GxB_unary_function", wrapper.address),
            result_type.gb_type,
            arg_type.gb_type,
        )

        return UnaryOp(func_name, arg_type.C, out[0])

    return inner
