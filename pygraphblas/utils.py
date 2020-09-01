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
