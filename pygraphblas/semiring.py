import sys
import re
import contextvars
from itertools import chain
from collections import defaultdict

from .base import lib, _gb_from_name

current_semiring = contextvars.ContextVar('current_semiring')


class Semiring:

    _auto_semirings = defaultdict(dict)

    __slots__ = ('name', 'semiring', 'token')

    def __init__(self, pls, mul, typ, semiring):
        self.name = '_'.join((pls, mul, typ))
        self.semiring = semiring
        self.token = None
        self.__class__._auto_semirings[pls+'_'+mul][_gb_from_name(typ)] = semiring

    def __enter__(self):
        self.token = current_semiring.set(self)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        current_semiring.reset(self.token)
        return False

    def get_semiring(self, operand1=None, operand2=None):
        return self.semiring


class AutoSemiring(Semiring):

    def __init__(self, name):
        self.name = name
        self.token = None

    def get_semiring(self, operand1=None, operand2=None):
        return Semiring._auto_semirings[self.name][operand1.gb_type]

__all__ = ['Semiring', 'AutoSemiring', 'current_semiring']

non_boolean_re = re.compile(
    '^GxB_(MIN|MAX|PLUS|TIMES)_'
    '(FIRST|SECOND|MIN|MAX|PLUS|MINUS|RMINUS|TIMES|DIV|RDIV|ISEQ|ISNE|ISGT|ISLT|ISGE|ISLE|LOR|LAND|LXOR)_'
    '(UINT8|UINT16|UINT32|UINT64|INT8|INT16|INT32|INT64|FP32|FP64)$')

boolean_re = re.compile(
    '^GxB_(LOR|LAND|LXOR|EQ)_'
    '(EQ|NE|GT|LT|GE|LE)_'
    '(UINT8|UINT16|UINT32|UINT64|INT8|INT16|INT32|INT64|FP32|FP64)$')

pure_bool_re = re.compile(
    '^GxB_(LOR|LAND|LXOR|EQ)_'
    '(FIRST|SECOND|LOR|LAND|LXOR|EQ|GT|LT|GE|LE)_(BOOL)$')

def semiring_group(reg):
    srs = []
    for n in filter(None, [reg.match(i) for i in dir(lib)]):
        pls, mul, typ = list(map(lambda g: g.lower(), n.groups()))
        srs.append(Semiring(pls, mul, typ, getattr(lib, n.string)))
    return srs

def build_semirings():
    this = sys.modules[__name__]
    for r in chain(semiring_group(non_boolean_re),
                   semiring_group(boolean_re),
                   semiring_group(pure_bool_re)):
        setattr(this, r.name, r)
        __all__.append(r.name)
    for name in Semiring._auto_semirings:
        sr = AutoSemiring(name)
        setattr(this, name, sr)
        __all__.append(name)
        
