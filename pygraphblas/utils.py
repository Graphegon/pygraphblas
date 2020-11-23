from .base import (
    lib,
    ffi,
    _check,
)

def get_version():
    version = ffi.new("unsigned int*")
    subversion = ffi.new("unsigned int*")
    _check(lib.GrB_getVersion(version, subversion))
    return (version[0], subversion[0])


def get_implementation_version():
    return (lib.GxB_IMPLEMENTATION_MAJOR,
            lib.GxB_IMPLEMENTATION_MINOR,
            lib.GxB_IMPLEMENTATION_SUB)


def get_spec_version():
    return (lib.GxB_SPEC_MAJOR,
            lib.GxB_SPEC_MINOR,
            lib.GxB_SPEC_SUB)
