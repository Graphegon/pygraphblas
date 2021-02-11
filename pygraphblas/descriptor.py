"""Contains all automatically generated Descriptors from CFFI.

"""
import contextvars
from .base import lib, ffi, _check

current_desc = contextvars.ContextVar("current_desc")


class Descriptor:
    """Wrapper class around GraphBLAS Descriptors.

    Descriptors "describe" the various options that can be used to
    control many aspects of graphblas operation.

    GraphBLAS Descriptors have a field and a value.  All of the common
    Descriptors necessary to use the GraphBLAS are available from the
    `pygraphblas.descriptor` module.

    Descriptors can be combined with the `&` operator.

    Descriptor | Description
    --- | ---
    `T0`      | Transpose First Argument
    `T1`      | Transpose Second Argument
    `T0T1`    | Transpose Both First and Second Argument
    `C`       | Complement Mask
    `CT1`     | C & T1
    `CT0`     | C & T0
    `CT0T1`   | T & T0 & T1
    `R`       | Replace Result
    `RT0`     | R & T0
    `RT1`     | R & T1
    `RT0T1`   | R & T0 & T1
    `RC`      | R & C
    `RCT0`    | R & C & T0
    `RCT1`    | R & C & T
    `RCT0T1`  | R & C & T0 & T1
    `S`       | Structural Mask
    `ST0`     | S & T0
    `ST1`     | S & T1
    `ST0T1`   | S & T0 & T1
    `RS`      | R & S
    `RST0`    | R & S & T0
    `RST1`    | R & S & T1
    `RST0T1`  | R & S & T0 & T1
    `RSC`     | R & S & C
    `RSCT0`   | R & S & C & T0
    `RSCT1`   | R & S & C & T1
    `RSCT0T1` |R & S & C & T0 & T1

    """

    __slots__ = ("field", "value", "_desc", "token", "name")

    def __init__(self, field, value, name=None):
        self.field = field
        self.value = value
        self._desc = ffi.new("GrB_Descriptor*")
        _check(lib.GrB_Descriptor_new(self._desc))
        self[field] = value
        self.token = None
        self.name = name

    def get_desc(self):
        return self._desc[0]

    def __enter__(self):
        self.token = current_desc.set(self)
        return self

    def __exit__(self, *errors):
        current_desc.reset(self.token)

    def __del__(self):
        if lib is not None:  # pragma: no cover
            _check(lib.GrB_Descriptor_free(self._desc))

    def __and__(self, other):
        d = Descriptor(self.field, self.value, self.name + other.name)
        d[other.field] = other.value
        return d

    def __contains__(self, other):
        return self[other.field] != lib.GxB_DEFAULT

    def __setitem__(self, field, value):
        _check(lib.GrB_Descriptor_set(self._desc[0], field, value))

    def __getitem__(self, field):
        val = ffi.new("GrB_Desc_Value*")
        _check(lib.GxB_Desc_get(self._desc[0], field, val))
        return val[0]

    def __eq__(self, other):
        for f in (
            lib.GrB_INP0,
            lib.GrB_INP1,
            lib.GrB_MASK,
            lib.GrB_OUTP,
            lib.GxB_DESCRIPTOR_NTHREADS,
            lib.GxB_DESCRIPTOR_CHUNK,
            lib.GxB_AxB_METHOD,
            lib.GxB_SORT,
        ):
            if self[f] != other[f]:
                return False
        return True

    def __repr__(self):
        return f"<Descriptor {self.name}>"


Default = Descriptor(lib.GrB_INP0, lib.GxB_DEFAULT, "Default")
T1 = Descriptor(lib.GrB_INP1, lib.GrB_TRAN, "T1")
T0 = Descriptor(lib.GrB_INP0, lib.GrB_TRAN, "T0")
T0T1 = T0 & T1

C = Descriptor(lib.GrB_MASK, lib.GrB_COMP, "C")
CT1 = C & T1
CT0 = C & T0
CT0T1 = C & T0 & T1

R = Descriptor(lib.GrB_OUTP, lib.GrB_REPLACE, "R")
RT0 = R & T0
RT1 = R & T1
RT0T1 = R & T0 & T1

RC = R & C
RCT0 = R & C & T0
RCT1 = R & C & T1
RCT0T1 = R & C & T0 & T1

S = Descriptor(lib.GrB_MASK, lib.GrB_STRUCTURE, "S")
ST1 = S & T1
ST0 = S & T0
ST0T1 = S & T0 & T1

RS = R & S
RST1 = R & S & T1
RST0 = R & S & T0
RST0T1 = R & S & T0 & T1

RSC = R & S & C
RSCT1 = R & S & C & T1
RSCT0 = R & S & C & T0
RSCT0T1 = R & S & C & T0 & T1

__all__ = [
    "Descriptor",
    "T1",
    "T0",
    "T0T1",
    "C",
    "CT1",
    "CT0",
    "CT0T1",
    "R",
    "RT0",
    "RT1",
    "RT0T1",
    "RC",
    "RCT0",
    "RCT1",
    "RCT0T1",
    "S",
    "ST1",
    "ST0",
    "ST0T1",
    "RS",
    "RST1",
    "RST0",
    "RST0T1",
    "RSC",
    "RSCT1",
    "RSCT0",
    "RSCT0T1",
]
