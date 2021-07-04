"""Contains all automatically generated BinaryOps from CFFI.

"""

__all__ = [
    "BinaryOp",
    "Accum",
    "current_binop",
    "current_accum",
    "binary_op",
]

import sys
import re
import contextvars
from itertools import chain
from collections import defaultdict
from functools import partial
import numba

from .base import lib, ffi, _check
from . import types

current_accum = contextvars.ContextVar("current_accum")
current_binop = contextvars.ContextVar("current_binop")


class BinaryOp:
    """Wrapper around GrB_BinaryOp."""

    def __init__(self, op, typ, binaryop, udt=None, boolean=False):
        if udt is not None:  # pragma: no cover
            o = ffi.new("GrB_BinaryOp*")
            udt = udt._gb_type
            lib.GrB_BinaryOp_new(
                o,
                ffi.cast("GxB_binary_function", binaryop.address),
                lib.GrB_BOOL if boolean else udt,
                udt,
                udt,
            )
            self.binaryop = o[0]
        else:
            self.binaryop = binaryop
            cls = getattr(types, typ)
            setattr(cls, op, self)
            setattr(cls, op.lower(), self)
        self.name = "_".join((op, typ))
        self.__doc__ = self.name
        self.token = None

    def __enter__(self):
        self.token = current_binop.set(self)
        return self

    def __exit__(self, *errors):  # pragma: nocover
        current_binop.reset(self.token)
        return False

    def get_binaryop(self, left=None, right=None):
        return self.binaryop

    def print(self, level=2, name="", f=sys.stdout):  # pragma: nocover
        """Print the matrix using `GxB_Matrix_fprint()`, by default to
        `sys.stdout`.

        Level 1: Short description
        Level 2: Short list, short numbers
        Level 3: Long list, short number
        Level 4: Short list, long numbers
        Level 5: Long list, long numbers

        """
        _check(lib.GxB_BinaryOp_fprint(self.binaryop, bytes(name, "utf8"), level, f))


class Accum:
    """Helper context manager to specify accumulator binary operator in
    overloaded operator contexts like `@`.  This disambiguates for
    methods like `Matrix.eadd` and `Matrix.emult` that can specify
    both a binary operators *and* a binary accumulator.

    See those methods and `Matrix.mxm` for examples.

    """

    __slots__ = ("binaryop", "token")

    def __init__(self, binaryop):
        self.binaryop = binaryop

    def __enter__(self):
        self.token = current_accum.set(self.binaryop)
        return self

    def __exit__(self, *errors):
        current_accum.reset(self.token)
        return False


grb_binop_re = re.compile(
    "^(GrB|GxB)_(FIRST|SECOND|MIN|MAX|PLUS|MINUS|RMINUS|TIMES|DIV|RDIV|"
    "FIRSTI|FIRSTI1|FIRSTJ|FIRSTJ1|SECONDI|SECONDI1|SECONDJ|SECONDJ1|"
    "PAIR|ANY|POW|EQ|NE|GT|LT|GE|LE|LOR|LAND|LXOR|BOR|BAND|BXOR|BXNOR|"
    "ATAN2|HYPOT|FMOD|REMAINDER|LDEXP|COPYSIGN|BGET|BSET|BCLR|BSHIFT|CMPLX)_"
    "(BOOL|UINT8|UINT16|UINT32|UINT64|INT8|INT16|INT32|INT64|FP32|FP64|FC32|FC64)$"
)

pure_bool_re = re.compile("^(GrB|GxB)_(LOR|LAND|LXOR)_(BOOL)$")


def binop_group(reg):
    srs = []
    for n in filter(None, [reg.match(i) for i in dir(lib)]):
        prefix, op, typ = n.groups()
        srs.append(BinaryOp(op, typ, getattr(lib, n.string)))
    return srs


def build_binaryops(__pdoc__):
    import tempfile

    this = sys.modules[__name__]
    for r in chain(binop_group(grb_binop_re), binop_group(pure_bool_re)):
        setattr(this, r.name, r)
        this.__all__.append(r.name)
        op, typ = r.name.split("_")
        f = tempfile.TemporaryFile()
        r.print(f=f)
        f.seek(0)
        __pdoc__[f"{typ}.{op}"] = f"""```{str(f.read(), 'utf8')}```"""


def binary_op(arg_type):
    """Decorator to jit-compile Python function into a GrB_BinaryOp
    object.

    >>> from random import uniform
    >>> from pygraphblas import Matrix, binary_op, types, gviz
    >>> @binary_op(types.FP64)
    ... def uniform(x, y):
    ...     return uniform(x, y)
    >>> A = Matrix.dense(types.FP64, 3, 3, fill=0)
    >>> B = A.dup()
    >>> with uniform:
    ...     A += 1

    Calling `A += 1` with the `uniform` binary operator is the same as
    calling `apply_second` with an `out` parameter:

    >>> B.apply_second(uniform, 1, out=B) is B
    True
    >>> ga = gviz.draw_matrix(A, scale=40,
    ...     filename='/docs/imgs/binary_op_A')
    >>> gb = gviz.draw_matrix(B, scale=40,
    ...     filename='/docs/imgs/binary_op_B')


    ![binary_op_A.png](../imgs/binary_op_A.png) ![binary_op_B.png](../imgs/binary_op_B.png)

    """

    def inner(func):
        func_name = func.__name__
        sig = numba.void(
            numba.types.CPointer(arg_type._numba_t),
            numba.types.CPointer(arg_type._numba_t),
            numba.types.CPointer(arg_type._numba_t),
        )
        jitfunc = numba.jit(func, nopython=True)

        @numba.cfunc(sig, nopython=True)
        def wrapper(z, x, y):  # pragma: no cover
            result = jitfunc(x[0], y[0])
            z[0] = result

        out = ffi.new("GrB_BinaryOp*")
        lib.GrB_BinaryOp_new(
            out,
            ffi.cast("GxB_binary_function", wrapper.address),
            arg_type._gb_type,
            arg_type._gb_type,
            arg_type._gb_type,
        )

        return BinaryOp(func_name, arg_type.__name__, out[0])

    return inner
