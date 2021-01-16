"""Contains all automatically generated Monoids from CFFI.

"""

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


__all__ = ["Monoid", "AutoMonoid", "current_monoid"]

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


def build_monoids():
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
