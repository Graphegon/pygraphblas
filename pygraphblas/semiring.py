import sys
import re
import contextvars
from itertools import chain

from .base import lib

current_semiring = contextvars.ContextVar('current_semiring')

class Semiring:

    __slots__ = ('name', 'semiring', 'token')

    def __init__(self, name, semiring):
        self.name = name
        self.semiring = semiring
        self.token = None

    def __enter__(self):
        self.token = current_semiring.set(self)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        current_semiring.reset(self.token)
        return False

class AutoSemiring(Semiring):

    __slots__ = ()
    

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
    '(FIRST|SECOND|LOR|LAND|LXOR|EQ|GT|LT|GE|LE)_BOOL$')

def semiring_group(reg):
    return [Semiring(n.string[4:].lower(), getattr(lib, n.string))
            for n in filter(None, [reg.match(i) for i in dir(lib)])]

def build_semirings():
    this = sys.modules[__name__]
    for r in chain(semiring_group(non_boolean_re),
                   semiring_group(boolean_re),
                   semiring_group(pure_bool_re)):
        setattr(this, r.name, r)
        __all__.append(r.name)
