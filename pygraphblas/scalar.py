from .base import (
    lib,
    ffi,
    _check,
    _check_no_val_key_error,
    _gb_from_type
)

from .type_funcs import build_scalar_type_funcs

NULL = ffi.NULL

__all__ = ['Scalar']

class Scalar:
    """GraphBLAS Scalar

    Used for now mostly for the matrix.select() functionality.

    """

    __slots__ = ('scalar', '_funcs')

    def __init__(self, s):
        self.scalar = s
        self._funcs = build_scalar_type_funcs(self.gb_type)

    def __del__(self):
        _check(lib.GxB_Scalar_free(self.scalar))

    def __len__(self):
        return self.nvals

    @classmethod
    def dup(cls, sca):
        """Create an duplicate Scalar from the given argument.

        """
        new_sca = ffi.new('GxB_Scalar*')
        _check(lib.GxB_Scalar_dup(new_sca, sca.scalar[0]))
        return cls(new_sca)

    @classmethod
    def from_type(cls, py_type):
        """Create an empty Scalar from the given type and size.

        """
        new_sca = ffi.new('GxB_Scalar*')
        gb_type = _gb_from_type(py_type)
        _check(lib.GxB_Scalar_new(new_sca, gb_type))
        return cls(new_sca)

    @classmethod
    def from_value(cls, value):
        """Create an empty Scalar from the given type and size.

        """
        new_sca = ffi.new('GxB_Scalar*')
        gb_type = _gb_from_type(type(value))
        _check(lib.GxB_Scalar_new(new_sca, gb_type))
        s = cls(new_sca)
        s[0] = value
        return s

    @property
    def gb_type(self):
        """Return the GraphBLAS low-level type object of the Scalar.

        """
        typ = ffi.new('GrB_Type*')
        _check(lib.GxB_Scalar_type(
            typ,
            self.scalar[0]))
        return typ[0]

    def clear(self):
        """Clear the scalar.

        """
        _check(lib.GxB_Scalar_clear(self.scalar[0]))

    def __getitem__(self, index):
        result = ffi.new(self._funcs.C + '*')
        _check_no_val_key_error(self._funcs.extractElement(
            result,
            self.scalar[0]
        ))
        return result[0]

    def __setitem__(self, index, value):
        _check(self._funcs.setElement(
            self.scalar[0],
            ffi.cast(self._funcs.C, value)
        ))
    
    @property
    def nvals(self):
        """Return the number of values in the scalar (0 or 1).

        """
        n = ffi.new('GrB_Index*')
        _check(lib.GxB_Scalar_nvals(n, self.scalar[0]))
        return n[0]

    def __bool__(self):
        return bool(self.nvals)
