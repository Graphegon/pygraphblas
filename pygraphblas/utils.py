from .base import (
    lib,
    ffi,
    _check,
)


def add_identity(M):
    from . import TransposeA

    d_out = M.reduce_vector()
    d_in = M.reduce_vector(desc=TransposeA)

    edges_added = 0
    if d_out.nvals < M.nrows or d_in.nvals < M.nrows:
        for i in range(M.nrows):
            if d_in.get(i) is None or d_out.get(i) is None:
                M.assign_scalar(M.type.one, i, i)
                edges_added += 1
    return edges_added


def get_version():
    version = ffi.new("unsigned int*")
    subversion = ffi.new("unsigned int*")
    _check(lib.GrB_getVersion(version, subversion))
    return (version[0], subversion[0])
