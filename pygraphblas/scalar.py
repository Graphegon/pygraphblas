from .base import (
    lib,
    ffi,
    NULL,
    _check,
    _check_no_val_key_error,
)

from .types import _gb_from_type

__all__ = ["Scalar"]


class Scalar:
    """GraphBLAS Scalar

    Used for now mostly for the matrix.select() functionality.

    """

    __slots__ = ("scalar", "_type")

    def __init__(self, s, typ):
        self.scalar = s
        self._type = typ

    def __del__(self):
        _check(lib.GxB_Scalar_free(self.scalar))

    def __len__(self):
        return self.nvals

    def dup(self):
        """Create an duplicate Scalar from the given argument.

        """
        new_sca = ffi.new("GxB_Scalar*")
        _check(lib.GxB_Scalar_dup(new_sca, self.scalar[0]))
        return self.__class__(new_sca, self._type)

    @classmethod
    def from_type(cls, typ):
        """Create an empty Scalar from the given type and size.

        """
        new_sca = ffi.new("GxB_Scalar*")
        _check(lib.GxB_Scalar_new(new_sca, typ.gb_type))
        return cls(new_sca, typ)

    @classmethod
    def from_value(cls, value):
        """Create an empty Scalar from the given type and size.

        """
        new_sca = ffi.new("GxB_Scalar*")
        typ = _gb_from_type(type(value))
        _check(lib.GxB_Scalar_new(new_sca, typ.gb_type))
        s = cls(new_sca, typ)
        s[0] = value
        return s

    @property
    def gb_type(self):
        """Return the GraphBLAS low-level type object of the Scalar.

        """
        typ = ffi.new("GrB_Type*")
        _check(lib.GxB_Scalar_type(typ, self.scalar[0]))
        return typ[0]

    def clear(self):
        """Clear the scalar.

        """
        _check(lib.GxB_Scalar_clear(self.scalar[0]))

    def __getitem__(self, index):
        result = ffi.new(self._type.C + "*")
        _check_no_val_key_error(
            self._type.Scalar_extractElement(result, self.scalar[0])
        )
        return result[0]

    def __setitem__(self, index, value):
        _check(
            self._type.Scalar_setElement(self.scalar[0], ffi.cast(self._type.C, value))
        )

    def wait(self):
        _check(lib.GxB_Scalar_wait(self.scalar))

    @property
    def nvals(self):
        """Return the number of values in the scalar (0 or 1).

        """
        n = ffi.new("GrB_Index*")
        _check(lib.GxB_Scalar_nvals(n, self.scalar[0]))
        return n[0]

    def __bool__(self):
        return bool(self.nvals)
