from .base import lib, ffi, _check

class Descriptor:

    def __init__(self, field, value):
        self.field = field
        self.value = value
        self.desc = ffi.new('GrB_Descriptor*')
        _check(lib.GrB_Descriptor_new(self.desc))
        self[field] = value

    def __del__(self):
        if lib is not None:
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
        val = ffi.new('GrB_Desc_Value*')
        _check(lib.GxB_Desc_get(self.desc[0], field, val))
        return val[0]


oooo = Default = Descriptor(lib.GrB_INP0, lib.GxB_DEFAULT)
tooo = TransposeA = Descriptor(lib.GrB_INP0, lib.GrB_TRAN)
otoo = TransposeB = Descriptor(lib.GrB_INP1, lib.GrB_TRAN)
ooco = ComplementMask = Descriptor(lib.GrB_MASK, lib.GrB_SCMP)
ooor = Replace = Descriptor(lib.GrB_OUTP, lib.GrB_REPLACE)

toco = tooo | ooco
toor = tooo | ooor
ttoo = tooo | otoo
oocr = ooco | ooor
otco = otoo | ooco
otor = otoo | ooor

tocr = tooo | ooco | ooor
ttco = tooo | otoo | ooco
ttor = tooo | otoo | ooor
otcr = otoo | ooco | ooor
ttcr = tooo | otoo | ooco | ooor

__all__ = [
    'Default', 'TransposeA', 'TransposeB', 'ComplementMask', 'Replace',
    ]
