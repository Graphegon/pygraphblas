import re, sys
from itertools import chain

from .base import lib

class BinaryOp:

    def __init__(self, name, binaryop):
        self.name = name
        self.binaryop = binaryop

    def __enter__(self):
        return self

    def __exit__(self, *errors):
        return False

grb_binop_re = re.compile('^GrB_(FIRST|SECOND|MIN|MAX|PLUS|MINUS|RMINUS|TIMES|DIV|RDIV|EQ|NE|GT|LT|GE|LE|LOR|LAND|LXOR)_(BOOL|INT64|FP64)$')

gxb_binop_re = re.compile('^GxB_(RMINUS|RDIV|ISEQ|ISNE|ISGT|ISLT|ISLE|ISGE)_(BOOL|INT64|FP64)$')

pure_bool_re = re.compile('^GrB_(LOR|LAND|LXOR)$')

def binop_group(reg):
    return [BinaryOp(n.string[4:].lower(), getattr(lib, n.string))
            for n in filter(None, [reg.match(i) for i in dir(lib)])]

def build_binaryops():
    this = sys.modules[__name__]
    for r in chain(binop_group(grb_binop_re), binop_group(gxb_binop_re), binop_group(pure_bool_re)):
        setattr(this, r.name, r)
