"""GraphBLAS Scalar (SuiteSparse Only)

"""
from .base import (
    lib,
    ffi,
    NULL,
    _check,
)

from .types import _gb_from_type

__all__ = ["Scalar"]


class Scalar:
    """GraphBLAS Scalar

    Used for now mostly for the `pygraphblas.Matrix.select`.

    """

    __slots__ = ("_scalar", "type")

    def __init__(self, s, typ):
        self._scalar = s
        self.type = typ

    def __del__(self):
        _check(lib.GxB_Scalar_free(self._scalar))

    def __len__(self):
        return self.nvals

    def dup(self):
        """Create an duplicate Scalar from the given argument."""
        new_sca = ffi.new("GxB_Scalar*")
        _check(lib.GxB_Scalar_dup(new_sca, self._scalar[0]))
        return self.__class__(new_sca, self.type)

    @classmethod
    def from_type(cls, typ):
        """Create an empty Scalar from the given type and size."""
        new_sca = ffi.new("GxB_Scalar*")
        _check(lib.GxB_Scalar_new(new_sca, typ._gb_type))
        return cls(new_sca, typ)

    @classmethod
    def from_value(cls, value):
        """Create an empty Scalar from the given type and size."""
        new_sca = ffi.new("GxB_Scalar*")
        typ = _gb_from_type(type(value))
        _check(lib.GxB_Scalar_new(new_sca, typ._gb_type))
        s = cls(new_sca, typ)
        s[0] = value
        return s

    @property
    def gb_type(self):
        """Return the GraphBLAS low-level type object of the Scalar."""
        return self.type._gb_type

    def clear(self):
        """Clear the scalar."""
        _check(lib.GxB_Scalar_clear(self._scalar[0]))

    def __getitem__(self, index):
        result = ffi.new(self.type._c_type + "*")
        _check(
            self.type._Scalar_extractElement(result, self._scalar[0]), raise_no_val=True
        )
        return result[0]

    def __setitem__(self, index, value):
        _check(
            self.type._Scalar_setElement(self._scalar[0], ffi.cast(self.type._c_type, value))
        )

    def wait(self):
        _check(lib.GxB_Scalar_wait(self._scalar))

    @property
    def nvals(self):
        """Return the number of values in the scalar (0 or 1)."""
        n = ffi.new("GrB_Index*")
        _check(lib.GxB_Scalar_nvals(n, self._scalar[0]))
        return n[0]

    def __bool__(self):
        return bool(self.nvals)
