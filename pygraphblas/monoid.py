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

__all__ = ["Monoid", "AutoMonoid", "current_monoid"]

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

    _auto_monoids = defaultdict(dict)

    __slots__ = ("name", "monoid", "token", "op", "type")

    def __init__(self, op, typ, monoid, udt=None, boolean=False):
        self.monoid = monoid
        self.__class__._auto_monoids[op + "_MONOID"][
            types.Type.gb_from_name(typ)
        ] = monoid
        cls = getattr(types, typ, None)
        if cls is not None:
            setattr(cls, op + "_MONOID", self)
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

    def get_monoid(self, left=None, right=None):
        return self.monoid


class AutoMonoid(Monoid):
    def __init__(self, name):
        self.name = name
        self.token = None

    def get_monoid(self, left=None, right=None):
        typ = types.promote(left, right)
        return Monoid._auto_monoids[self.name][typ.gb_type]


gxb_monoid_re = re.compile(
    "^GxB_(MIN|MAX|PLUS|TIMES|ANY|BOR|BAND|BXOR|BXNOR)_"
    "(UINT8|UINT16|UINT32|UINT64|INT8|INT16|INT32|INT64|FP32|FP64)_MONOID$"
)

grb_monoid_re = re.compile(
    "^GrB_(MIN|MAX|PLUS|TIMES)_MONOID_"
    "(UINT8|UINT16|UINT32|UINT64|INT8|INT16|INT32|INT64|FP32|FP64)$"
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
    this = sys.modules[__name__]
    for r in chain(
        monoid_group(gxb_monoid_re),
        monoid_group(grb_monoid_re),
        monoid_group(pure_bool_re),
        monoid_group(pure_bool_re_v13),
    ):
        setattr(this, r.name, r)
    for name in Monoid._auto_monoids:
        bo = AutoMonoid(name)
        setattr(this, name, bo)
