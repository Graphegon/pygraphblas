"""Contains all automatically generated Monoids from CFFI.

The scalar addition of conventional matrix multiplication is replaced
with a *monoid*.  A monoid is an associative and commutative binary
operator `z=f(x,y)` where all three domains are the same (the
types of `x`, `y`, and `z`), and where the operator has
an identity value `id` such that `f(x,id)=f(id,x)=x`.
Performing matrix multiplication with a semiring uses a monoid in
place of the `add` operator, scalar addition being just one of many
possible monoids.  The identity value of addition is zero, since
$x+0=0+x=x$.  GraphBLAS includes many built-in operators suitable for
use as a monoid: min (with an identity value of positive infinity),
max (whose identity is negative infinity), add (identity is zero),
multiply (with an identity of one), four logical operators: AND, OR,
exclusive-OR, and Boolean equality (XNOR), four bitwise operators
(AND, OR, XOR, and XNOR), and the ANY operator.  User-created monoids
can be defined with any associative and commutative operator that has
an identity value.
"""

__all__ = ["Monoid", "current_monoid"]

import os
import sys
import re
import contextvars
from itertools import chain
from collections import defaultdict

from .base import lib, ffi, _check
from .binaryop import BinaryOp
from . import types

current_monoid = contextvars.ContextVar("current_monoid")


class Monoid:

    __slots__ = ("name", "monoid", "token", "op", "type")

    def __init__(self, op, typ, monoid, udt=None, boolean=False):
        self.monoid = monoid
        cls = getattr(types, typ, None)
        if cls is not None:
            setattr(cls, op + "_MONOID", self)
            setattr(cls, op.lower() + "_monoid", self)
            types.__pdoc__[f"{typ}.{op}_MONOID"] = f"UnaryOp {typ}.{op}_MONOID"
        self.op = op
        self.type = typ
        self.name = "_".join((op, typ, "monoid"))
        self.token = None

    def __enter__(self):
        self.token = current_monoid.set(self)
        return self

    def __exit__(self, *errors):
        current_monoid.reset(self.token)
        return False

    def __call__(self, A, B, *args, **kwargs):
        return A.eadd(B, self, *args, **kwargs)

    def get_op(self):
        return self.monoid

    def print(self, level=2, name="", f=sys.stdout):  # pragma: nocover
        """Print the matrix using `GxB_Matrix_fprint()`, by default to
        `sys.stdout`.

        Level 1: Short description
        Level 2: Short list, short numbers
        Level 3: Long list, short number
        Level 4: Short list, long numbers
        Level 5: Long list, long numbers

        """
        _check(lib.GxB_Monoid_fprint(self.monoid, bytes(name, "utf8"), level, f))


gxb_monoid_re = re.compile(
    "^GxB_(MIN|MAX|PLUS|TIMES|ANY|BOR|BAND|BXOR|BXNOR)_"
    "(UINT8|UINT16|UINT32|UINT64|INT8|INT16|INT32|INT64|FP32|FP64|FC32|FC64)_MONOID$"
)

grb_monoid_re = re.compile(
    "^GrB_(MIN|MAX|PLUS|TIMES)_MONOID_"
    "(UINT8|UINT16|UINT32|UINT64|INT8|INT16|INT32|INT64|FP32|FP64|FC32|FC64)$"
)

pure_bool_re = re.compile("^GxB_(ANY|LOR|LAND|LXOR|LXNOR|EQ)_(BOOL)_MONOID$")
pure_bool_re_v13 = re.compile("^GrB_(LOR|LAND|LXOR|LXNOR)_MONOID_(BOOL)$")


def monoid_group(reg):
    srs = []
    for n in filter(None, [reg.match(i) for i in dir(lib)]):
        op, typ = n.groups()
        m = Monoid(op, typ, getattr(lib, n.string))
        srs.append(m)
    return srs


def build_monoids(__pdoc__):
    import tempfile

    this = sys.modules[__name__]
    for r in chain(
        monoid_group(gxb_monoid_re),
        monoid_group(grb_monoid_re),
        monoid_group(pure_bool_re),
        monoid_group(pure_bool_re_v13),
    ):
        setattr(this, r.name, r)
        f = tempfile.TemporaryFile()
        r.print(f=f)
        f.seek(0)
        this.__all__.append(r.name)
        op, typ, _ = r.name.split("_")
        __pdoc__[f"{typ}.{op}_MONOID"] = f"""```{str(f.read(), 'utf8')}```"""
