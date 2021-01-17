"""Contains all automatically generated Descriptors from CFFI.

"""
import contextvars
from .base import lib, ffi, _check

current_desc = contextvars.ContextVar("current_desc")


class Descriptor:
    """Wrapper class around pre-defined GraphBLAS Descriptors."""

    __slots__ = ("field", "value", "desc", "token")

    def __init__(self, field, value):
        self.field = field
        self.value = value
        self.desc = ffi.new("GrB_Descriptor*")
        _check(lib.GrB_Descriptor_new(self.desc))
        self[field] = value
        self.token = None

    def __enter__(self):
        self.token = current_desc.set(self)
        return self

    def __exit__(self, *errors):
        current_desc.reset(self.token)

    def __del__(self):
        if lib is not None:  # pragma: no cover
            _check(lib.GrB_Descriptor_free(self.desc))

    def __or__(self, other):
        d = Descriptor(self.field, self.value)
        d[other.field] = other.value
        return d

    def __contains__(self, other):
        return self[other.field] != lib.GxB_DEFAULT

    def __setitem__(self, field, value):
        _check(lib.GrB_Descriptor_set(self.desc[0], field, value))

    def __getitem__(self, field):
        val = ffi.new("GrB_Desc_Value*")
        _check(lib.GxB_Desc_get(self.desc[0], field, val))
        return val[0]

    def __eq__(self, other):
        for f in (lib.GrB_INP0, lib.GrB_INP1, lib.GrB_MASK, lib.GrB_OUTP):
            if self[f] != other[f]:
                return False
        return True


# three sets of descriptor names here
# lagraph_name = new_name = VerboseName

oooo = Default = Descriptor(lib.GrB_INP0, lib.GxB_DEFAULT)
otoo = T1 = TransposeB = Descriptor(lib.GrB_INP1, lib.GrB_TRAN)
tooo = T0 = TransposeA = Descriptor(lib.GrB_INP0, lib.GrB_TRAN)
ttoo = T0T1 = TransposeATransposeB = T0 | T1

ooco = C = ComplementMask = Descriptor(lib.GrB_MASK, lib.GrB_COMP)
otco = CT1 = TransposeBComplementMask = C | T1
toco = CT0 = TransposeAComplementMask = C | T0
ttco = CT0T1 = TransposeATransposeBComplementMask = C | T0 | T1

ooor = R = Replace = Descriptor(lib.GrB_OUTP, lib.GrB_REPLACE)
toor = RT0 = TransposeAReplace = R | T0
otor = RT1 = TransposeBReplace = R | T1
ttor = RT0T1 = R | T0 | T1

oocr = RC = ComplementMaskReplace = R | C
tocr = RCT0 = R | C | T0
otcr = RCT1 = R | C | T1
ttcr = RCT0T1 = R | C | T0 | T1

# STRUCTURAL is new so it doesnt have an lagraph naming scheme or
# verbose names due to their silly length.

oosoo = S = Descriptor(lib.GrB_MASK, lib.GrB_STRUCTURE)
otsoo = ST1 = S | T1
tosoo = ST0 = S | T0
ttsoo = ST0T1 = S | T0 | T1

oosor = RS = R | S
otsor = RST1 = R | S | T1
tosor = RST0 = R | S | T0
ttsor = RST0T1 = R | S | T0 | T1

ooscr = RSC = R | S | C
otscr = RSCT1 = R | S | C | T1
toscr = RSCT0 = R | S | C | T0
ttscr = RSCT0T1 = R | S | C | T0 | T1

__all__ = [
    "TransposeA",
    "TransposeB",
    "ComplementMask",
    "Replace",
    "TransposeAComplementMask",
    "TransposeAReplace",
    "TransposeBComplementMask",
    "TransposeBReplace",
    "TransposeATransposeB",
    "ComplementMaskReplace",
    "oooo",
    "otoo",
    "tooo",
    "ttoo",
    "ooco",
    "otco",
    "toco",
    "ttco",
    "ooor",
    "toor",
    "otor",
    "ttor",
    "oocr",
    "tocr",
    "otcr",
    "ttcr",
    "oosoo",
    "otsoo",
    "tosoo",
    "ttsoo",
    "oosor",
    "otsor",
    "tosor",
    "ttsor",
    "ooscr",
    "otscr",
    "toscr",
    "ttscr",
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
