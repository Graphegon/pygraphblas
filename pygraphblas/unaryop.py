import re, sys
from itertools import chain

from .base import lib

class UnaryOp:

    def __init__(self, name, unaryop):
        self.name = name
        self.unaryop = unaryop

    def __enter__(self):
        return self

    def __exit__(self, *errors):
        return False

__all__ = ['UnaryOp']

grb_uop_re = re.compile('^GrB_(IDENTITY|AINV|MINV|LNOT|ONE|ABS)_(BOOL|INT64|FP64)$')

bool_uop_re = re.compile('^GrB_LNOT$')

def uop_group(reg):
    return [UnaryOp(n.string[4:].lower(), getattr(lib, n.string))
            for n in filter(None, [reg.match(i) for i in dir(lib)])]

def build_unaryops():
    this = sys.modules[__name__]
    for r in chain(uop_group(grb_uop_re), uop_group(bool_uop_re)):
        setattr(this, r.name, r)
        __all__.append(r.name)
