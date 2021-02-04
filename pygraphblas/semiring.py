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

__all__ = ["Semiring", "current_semiring"]


class Semiring:

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
            cls = getattr(types, typ)
            setattr(cls, name, self)
            types.__pdoc__[f"{typ}.{pls}_{mul}"] = f"UnaryOp {typ}.{pls}_{mul}"

    def __enter__(self):
        self.token = current_semiring.set(self)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        current_semiring.reset(self.token)
        return False

    def get_semiring(self, typ=None):
        return self.semiring

    @property
    def ztype(self):
        return types.get_semiring_ztype(self.semiring)

    def print(self, level=2, name="", f=sys.stdout):  # pragma: nocover
        """Print the matrix using `GxB_Matrix_fprint()`, by default to
        `sys.stdout`.

        Level 1: Short description
        Level 2: Short list, short numbers
        Level 3: Long list, short number
        Level 4: Short list, long numbers
        Level 5: Long list, long numbers

        """
        _check(lib.GxB_Semiring_fprint(self.semiring, bytes(name, "utf8"), level, f))


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


def semiring_template(r):  # pragma: nocover
    from .matrix import Matrix

    if r.ztype in (types.FC32, types.FC64):
        return f"Semiring {r.name}"
    A = Matrix.from_lists([0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0])
    B = A.dup()
    if r.ztype is types.BOOL:
        A = A.pattern()
        B = B.pattern()
    C = A.mxm(B, semiring=r)
    return f"""\
Semiring {r.name}

<table>
<tr>
<td>{A.to_html_table('A')}</td><td> {B.to_html_table('B')}</td><td> {C.to_html_table('A @ B')}</td>
</tr>
</table>
"""


def build_semirings(__pdoc__):
    import tempfile

    this = sys.modules[__name__]
    for r in chain(
        semiring_group(non_boolean_re),
        semiring_group(boolean_re),
        semiring_group(pure_bool_re),
        semiring_group(bitwise_re),
        semiring_group(complex_re),
    ):
        setattr(this, r.name, r)
        pls, mul, typ = r.name.split("_")
        f = tempfile.TemporaryFile()
        r.print(f=f)
        f.seek(0)
        __pdoc__[f"{typ}.{pls}_{mul}"] = f"""```{str(f.read(), 'utf8')}```"""
