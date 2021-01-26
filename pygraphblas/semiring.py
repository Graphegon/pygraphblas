"""Contains all automatically generated Semirings from CFFI.

This documentation does not show all the semirings in this module
because of the sheer number of them (over 1700).  Please see the
SuiteSparse User Guide for more information on the semirings usable in
The GraphBLAS.

All the standard and extension semirings that comes with SuiteSparse
are represented by objects in this module.  For example
`pygraphblas.semiring.PLUS_TIMES_INT64`.

"""

import sys
import re
import contextvars
from itertools import chain
from collections import defaultdict

from .base import lib, ffi, _check
from .monoid import Monoid
from . import types

current_semiring = contextvars.ContextVar("current_semiring")

__all__ = ["Semiring", "AutoSemiring", "current_semiring"]


class Semiring:

    _auto_semirings = defaultdict(dict)

    __slots__ = ("name", "semiring", "token", "pls", "mul", "type")

    def __init__(self, pls, mul, typ, semiring, udt=None):
        self.pls = pls
        self.mul = mul
        self.type = typ
        self.name = "_".join((pls, mul, typ))
        self.semiring = semiring
        self.token = None
        name = pls + "_" + mul
        if udt is None:
            self.__class__._auto_semirings[name][
                types.Type.gb_from_name(typ)
            ] = semiring
            cls = getattr(types, typ)
            setattr(cls, name, self)

    def __enter__(self):
        self.token = current_semiring.set(self)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        current_semiring.reset(self.token)
        return False

    def get_semiring(self, typ):
        return self.semiring


class AutoSemiring(Semiring):
    def __init__(self, name):
        self.name = name
        self.token = None

    def get_semiring(self, typ):
        return Semiring._auto_semirings[self.name][typ.gb_type]


non_boolean_re = re.compile(
    "^(GxB|GrB)_(MIN|MAX|PLUS|TIMES|ANY)_"
    "(FIRST|FIRSTI|FIRSTJ|FIRSTI1|FIRSTJ1|SECOND|SECONDI|SECONDJ|SECONDI1|SECONDJ1|MIN|MAX|PLUS|MINUS|RMINUS|TIMES|DIV|RDIV|ISEQ|ISNE|"
    "ISGT|ISLT|ISGE|ISLE|LOR|LAND|LXOR|PAIR)_"
    "(UINT8|UINT16|UINT32|UINT64|INT8|INT16|INT32|INT64|FP32|FP64)$"
)

boolean_re = re.compile(
    "^(GxB|GrB)_(LOR|LAND|LXOR|EQ|ANY)_"
    "(EQ|NE|GT|LT|GE|LE)_"
    "(UINT8|UINT16|UINT32|UINT64|INT8|INT16|INT32|INT64|FP32|FP64)$"
)

pure_bool_re = re.compile(
    "^(GxB|GrB)_(LOR|LAND|LXOR|EQ|ANY)_"
    "(FIRST|SECOND|LOR|LAND|LXOR|EQ|GT|LT|GE|LE|PAIR)_(BOOL)$"
)

complex_re = re.compile(
    "^(GxB|GrB)_(PLUS|TIMES|ANY)_"
    "(FIRST|SECOND|PLUS|MINUS|RMINUS|TIMES|DIV|RDIV|PAIR)_"
    "(FC32|FC64)$"
)

bitwise_re = re.compile(
    "^(GxB|GrB)_(BOR|BAND|BXOR|BXNOR)_"
    "(BOR|BAND|BXOR|BXNOR)_"
    "(UINT8|UINT16|UINT32|UINT64)$"
)


def semiring_group(reg):
    srs = []
    for n in filter(None, [reg.match(i) for i in dir(lib)]):
        prefix, pls, mul, typ = n.groups()
        srs.append(Semiring(pls, mul, typ, getattr(lib, n.string)))
    return srs


def build_semirings():
    this = sys.modules[__name__]
    for r in chain(
        semiring_group(non_boolean_re),
        semiring_group(boolean_re),
        semiring_group(pure_bool_re),
        semiring_group(bitwise_re),
        semiring_group(complex_re),
    ):
        setattr(this, r.name, r)
    for name in Semiring._auto_semirings:
        sr = AutoSemiring(name)
        setattr(this, name, sr)
