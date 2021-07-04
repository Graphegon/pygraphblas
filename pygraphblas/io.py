from . import ffi, lib, types
from .base import _check
from pathlib import Path
import ctypes
from cffi import FFI


stdffi = FFI()
stdffi.cdef("""
void *malloc(size_t size);
""")
stdlib = stdffi.dlopen(None)

# When "packing" a matrix the owner of the memory buffer is transfered
# to SuiteSparse, which then becomes responsible for freeing it.  cffi
# wisely does not allow you to do this without declaring and calling
# malloc directly.  When SuiteSparse moves over to a more formal
# memory manager with the cuda work, this will likely change and have
# to be replaceable with a allocator common to numpy, cuda, and here.
# Maybe PyDataMem_NEW?
malloc = stdlib.malloc

GRB_HEADER_LEN=512

header_template = """\
SuiteSparse:GraphBLAS matrix
v{major}.{minor}.{sub} (pygraphblas {pyversion})
nrows:   {nrows}
ncols:   {ncols}
nvec:    {nvec}
nvals:   {nvals}
format:  {format}
size:    {size}
type:    {type}
iso:     {iso}
jumbled: {iso}
{comments}
"""

sizeof = ffi.sizeof
ffinew = ffi.new
buff = ffi.buffer
frombuff = ffi.from_buffer
Isize = ffi.sizeof('GrB_Index')
    
_ss_typecodes = {
    types.BOOL: 0,
    types.INT8: 1,
    types.INT16: 2,
    types.INT32: 3,
    types.INT64: 4,
    types.UINT8: 5,
    types.UINT16: 6,
    types.UINT32: 7,
    types.UINT64: 8,
    types.FP32: 9,
    types.FP64: 10,
    types.FC32: 11,
    types.FC64: 12,
    }

_ss_codetypes = {v: k for k, v in _ss_typecodes.items()}


def readinto_new_buffer(f, typ, size):
    buff = ffi.cast(typ, malloc(size))
    f.readinto(ffi.buffer(buff, size))
    return buff

def matrix_binwrite(A, filename, comments=None, compression=None):
    from . import get_version

    A.wait()
    ffinew = ffi.new

    Ap = ffinew('GrB_Index**')
    Ai = ffinew('GrB_Index**')
    Ah = ffinew('GrB_Index**')
    Ax = ffinew('void**')
    Ab = ffinew('int8_t**')

    Ap_size = ffinew('GrB_Index*')
    Ai_size = ffinew('GrB_Index*')
    Ah_size = ffinew('GrB_Index*')
    Ax_size = ffinew('GrB_Index*')
    Ab_size = ffinew('GrB_Index*')

    nvec  = ffinew('GrB_Index*')
    nrows = ffinew('GrB_Index*', A.nrows)
    ncols = ffinew('GrB_Index*', A.ncols)
    nvals = ffinew('GrB_Index*', A.nvals)

    is_iso     = ffinew('bool*')
    is_jumbled = ffinew('bool*')

    nonempty     = ffinew('int64_t*',          -1)
    typecode     = ffinew('int32_t*',          _ss_typecodes[A.type])
    typesize     = ffinew('int32_t*',          A.type.size)
    hyper_switch = ffinew('double*',           A.hyper_switch)
    format       = ffinew('GxB_Format_Value*', A.format)
    status       = ffinew('int32_t*',          A.sparsity_status)
    
    by_row = format[0] == lib.GxB_BY_ROW
    by_col = format[0] == lib.GxB_BY_COL

    is_hyper  = status[0] == lib.GxB_HYPERSPARSE
    is_sparse = status[0] == lib.GxB_SPARSE
    is_bitmap = status[0] == lib.GxB_BITMAP
    is_full   = status[0] == lib.GxB_FULL

    if by_col and is_hyper:
        A._check(lib.GxB_Matrix_unpack_HyperCSC (
            A._matrix[0],
            Ap, Ah, Ai, Ax, Ap_size, Ah_size, Ai_size, Ax_size,
            is_iso, nvec, is_jumbled, ffi.NULL))
        fmt_string = "HCSC"

    elif by_row and is_hyper:
        A._check(lib.GxB_Matrix_unpack_HyperCSR (
            A._matrix[0],
            Ap, Ah, Ai, Ax, Ap_size, Ah_size, Ai_size, Ax_size,
            is_iso, nvec, is_jumbled, ffi.NULL))
        fmt_string = "HCSR"

    elif by_col and is_sparse:
        A._check(lib.GxB_Matrix_unpack_CSC (
            A._matrix[0],
            Ap, Ai, Ax, Ap_size, Ai_size, Ax_size,
            is_iso, is_jumbled, ffi.NULL))
        nvec[0] = ncols[0]
        fmt_string = "CSC"

    elif by_row and is_sparse:
        A._check(lib.GxB_Matrix_unpack_CSR (
            A._matrix[0],
            Ap, Ai, Ax, Ap_size, Ai_size, Ax_size,
            is_iso, is_jumbled, ffi.NULL))
        nvec[0] = nrows[0]
        fmt_string = "CSR"

    elif by_col and is_bitmap:
        A._check(lib.GxB_Matrix_unpack_BitmapC(
            A._matrix[0],
            Ab, Ax, Ab_size, Ax_size, is_iso, nvals, ffi.NULL
            ))
        nvec[0] = ncols[0]
        fmt_string = "BITMAPC"

    elif by_row and is_bitmap:
        A._check(lib.GxB_Matrix_unpack_BitmapR(
            A._matrix[0],
            Ab, Ax, Ab_size, Ax_size, is_iso, nvals, ffi.NULL
            ))
        nvec[0] = nrows[0]
        fmt_string = "BITMAPR"

    elif by_col and is_full:
        A._check(lib.GxB_Matrix_unpack_FullC(
            A._matrix[0],
            Ax, Ax_size, is_iso, ffi.NULL
            ))
        nvec[0] = ncols[0]
        fmt_string = "FULLC"

    elif by_row and is_full:
        A._check(lib.GxB_Matrix_unpack_FullR(
            A._matrix[0],
            Ax, Ax_size, is_iso, ffi.NULL
            ))
        nvec[0] = nrows[0]
        fmt_string = "FULLR"

    else:
        raise TypeError(f'Unknown Matrix format {format[0]}')

    vars = dict(
        major     = lib.GxB_IMPLEMENTATION_MAJOR,
        minor     = lib.GxB_IMPLEMENTATION_MINOR,
        sub       = lib.GxB_IMPLEMENTATION_SUB,
        pyversion = get_version(),
        nrows     = nrows[0],
        ncols     = ncols[0],
        nvals     = nvals[0],
        nvec      = nvec[0],
        format    = fmt_string,
        size      = A.type.size,
        type      = A.type.GrB_name,
        iso       = is_iso[0],
        jumbled   = is_jumbled[0],
        comments  = comments,
    )
    header_content = header_template.format(**vars)
    header = f'{header_content: <{GRB_HEADER_LEN}}'.encode('ascii')

    if compression is None:
        opener = Path.open
    elif compression == 'gzip':
        import gzip
        opener = gzip.open

    with opener(filename, 'wb') as f:
        fwrite = f.write
        fwrite(header)
        fwrite(buff(format,       sizeof('GxB_Format_Value')))
        fwrite(buff(status,       sizeof('int32_t')))
        fwrite(buff(hyper_switch, sizeof('double')))
        fwrite(buff(nrows,        Isize))
        fwrite(buff(ncols,        Isize))
        fwrite(buff(nonempty,     sizeof('int64_t')))
        fwrite(buff(nvec,         Isize))
        fwrite(buff(nvals,        Isize))
        fwrite(buff(typecode,     sizeof('int32_t')))
        fwrite(buff(typesize,     sizeof('size_t')))
        fwrite(buff(is_iso,       sizeof('bool')))
        fwrite(buff(is_jumbled,   sizeof('bool')))

        if is_hyper:
            Ap_size[0] = (nvec[0] + 1) * Isize
            Ah_size[0] = nvec[0] * Isize
            Ai_size[0] = nvals[0] * Isize
            Ax_size[0] = nvals[0] * typesize[0]
            fwrite(buff(Ap[0], Ap_size[0]))
            fwrite(buff(Ah[0], Ah_size[0]))
            fwrite(buff(Ai[0], Ai_size[0]))
        elif is_sparse:
            Ap_size[0] = (nvec[0] + 1) * Isize
            Ai_size[0] = nvals[0] * Isize
            Ax_size[0] = nvals[0] * typesize[0]
            fwrite(buff(Ap[0], Ap_size[0]))
            fwrite(buff(Ai[0], Ai_size[0]))
        elif is_bitmap:
            Ab_size[0] = nrows[0] * ncols[0] * ffi.sizeof('int8_t')
            Ax_size[0] = nrows[0] * ncols[0] * typesize[0]
            fwrite(buff(Ab[0], Ab_size[0]))
            
        fwrite(buff(Ax[0], Ax_size[0]))

    if by_col and is_hyper:
        A._check(lib.GxB_Matrix_pack_HyperCSC (
            A._matrix[0],
            Ap, Ah, Ai, Ax, Ap_size[0], Ah_size[0], Ai_size[0], Ax_size[0],
            is_iso[0], nvec[0], is_jumbled[0], ffi.NULL))

    elif by_row and is_hyper:
        A._check(lib.GxB_Matrix_pack_HyperCSR (
            A._matrix[0],
            Ap, Ah, Ai, Ax, Ap_size[0], Ah_size[0], Ai_size[0], Ax_size[0],
            is_iso[0], nvec[0], is_jumbled[0], ffi.NULL))

    elif by_col and is_sparse:
        A._check(lib.GxB_Matrix_pack_CSC (
            A._matrix[0],
            Ap, Ai, Ax, Ap_size[0], Ai_size[0], Ax_size[0],
            is_iso[0], is_jumbled[0], ffi.NULL))

    elif by_row and is_sparse:
        A._check(lib.GxB_Matrix_pack_CSR (
            A._matrix[0],
            Ap, Ai, Ax, Ap_size[0], Ai_size[0], Ax_size[0],
            is_iso[0], is_jumbled[0], ffi.NULL))

    elif by_col and is_bitmap:
        A._check(lib.GxB_Matrix_pack_BitmapC(
            A._matrix[0],
            Ab, Ax, Ab_size[0], Ax_size[0], is_iso[0], nvals[0], ffi.NULL
            ))

    elif by_row and is_bitmap:
        A._check(lib.GxB_Matrix_pack_BitmapR(
            A._matrix[0],
            Ab, Ax, Ab_size[0], Ax_size[0], is_iso[0], nvals[0], ffi.NULL
            ))

    elif by_col and is_full:
        A._check(lib.GxB_Matrix_pack_FullC(
            A._matrix[0],
            Ax, Ax_size[0], is_iso[0], ffi.NULL
            ))

    elif by_row and is_full:
        A._check(lib.GxB_Matrix_pack_FullR(
            A._matrix[0],
            Ax, Ax_size[0], is_iso[0], ffi.NULL
            ))
    else:
        raise TypeError('This should hever happen')
    A.hyper_switch = hyper_switch[0]

def matrix_binread(filename, compression=None):

    if compression is None:
        opener = Path.open
    elif compression == 'gzip':
        import gzip
        opener = gzip.open

    with opener(filename, 'rb') as f:
        fread = f.read

        header       = fread(GRB_HEADER_LEN)
        format       = frombuff('GxB_Format_Value*', fread(sizeof('GxB_Format_Value')))
        status       = frombuff('int32_t*',          fread(sizeof('int32_t')))
        hyper_switch = frombuff('double*',           fread(sizeof('double')))
        nrows        = frombuff('GrB_Index*',        fread(Isize))
        ncols        = frombuff('GrB_Index*',        fread(Isize))
        nonempty     = frombuff('int64_t*',          fread(sizeof('int64_t')))
        nvec         = frombuff('GrB_Index*',        fread(Isize))
        nvals        = frombuff('GrB_Index*',        fread(Isize))
        typecode     = frombuff('int32_t*',          fread(sizeof('int32_t')))
        typesize     = frombuff('size_t*',           fread(sizeof('size_t')))
        is_iso       = frombuff('bool*',             fread(sizeof('bool')))
        is_jumbled   = frombuff('bool*',             fread(sizeof('bool')))

        by_row  = format[0] == lib.GxB_BY_ROW
        by_col  = format[0] == lib.GxB_BY_COL

        is_hyper  = status[0] == lib.GxB_HYPERSPARSE
        is_sparse = status[0] == lib.GxB_SPARSE
        is_bitmap = status[0] == lib.GxB_BITMAP
        is_full   = status[0] == lib.GxB_FULL

        atype = _ss_codetypes[typecode[0]]

        Ap = ffinew('GrB_Index**')
        Ai = ffinew('GrB_Index**')
        Ah = ffinew('GrB_Index**')
        Ax = ffinew('void**')
        Ab = ffinew('int8_t**')

        Ap_size = ffinew('GrB_Index*')
        Ai_size = ffinew('GrB_Index*')
        Ah_size = ffinew('GrB_Index*')
        Ax_size = ffinew('GrB_Index*')
        Ab_size = ffinew('GrB_Index*')

        if is_hyper:
            Ap_size[0] = (nvec[0] + 1) * Isize
            Ah_size[0] = nvec[0] * Isize
            Ai_size[0] = nvals[0] * Isize
            Ax_size[0] = nvals[0] * typesize[0]

            Ap[0] = readinto_new_buffer(f, 'GrB_Index*', Ap_size[0])
            Ah[0] = readinto_new_buffer(f, 'GrB_Index*', Ah_size[0])
            Ai[0] = readinto_new_buffer(f, 'GrB_Index*', Ai_size[0])
        elif is_sparse:
            Ap_size[0] = (nvec[0] + 1) * Isize
            Ai_size[0] = nvals[0] * Isize
            Ax_size[0] = nvals[0] * typesize[0]
            Ap[0] = readinto_new_buffer(f, 'GrB_Index*', Ap_size[0])
            Ai[0] = readinto_new_buffer(f, 'GrB_Index*', Ai_size[0])
        elif is_bitmap:
            Ab_size[0] = nrows[0] * ncols[0] * ffi.sizeof('int8_t')
            Ax_size[0] = nrows[0] * ncols[0] * typesize[0]
            Ab[0] = readinto_new_buffer(f, 'int8_t*', Ab_size[0])
        elif is_full:
            Ax_size[0] = nrows[0] * ncols[0] * typesize[0]
            
        Ax[0] = readinto_new_buffer(f, 'uint8_t*', Ax_size[0])

        _A = ffi.new("GrB_Matrix*")
        _check(lib.GrB_Matrix_new(_A, atype._gb_type, nrows[0], ncols[0]))
        
        if by_col and is_hyper:
            _check(lib.GxB_Matrix_pack_HyperCSC (
                _A[0],
                Ap, Ah, Ai, Ax, Ap_size[0], Ah_size[0], Ai_size[0], Ax_size[0],
                is_iso[0], nvec[0], is_jumbled[0], ffi.NULL))

        elif by_row and is_hyper:
            _check(lib.GxB_Matrix_pack_HyperCSR (
                _A[0],
                Ap, Ah, Ai, Ax, Ap_size[0], Ah_size[0], Ai_size[0], Ax_size[0],
                is_iso[0], nvec[0], is_jumbled[0], ffi.NULL))

        elif by_col and is_sparse:
            _check(lib.GxB_Matrix_pack_CSC (
                _A[0],
                Ap, Ai, Ax, Ap_size[0], Ai_size[0], Ax_size[0],
                is_iso[0], is_jumbled[0], ffi.NULL))

        elif by_row and is_sparse:
            _check(lib.GxB_Matrix_pack_CSR (
                _A[0],
                Ap, Ai, Ax, Ap_size[0], Ai_size[0], Ax_size[0],
                is_iso[0], is_jumbled[0], ffi.NULL))

        elif by_col and is_bitmap:
            _check(lib.GxB_Matrix_pack_BitmapC(
                _A[0],
                Ab, Ax, Ab_size[0], Ax_size[0], is_iso[0], nvals[0], ffi.NULL
                ))

        elif by_row and is_bitmap:
            _check(lib.GxB_Matrix_pack_BitmapR(
                _A[0],
                Ab, Ax, Ab_size[0], Ax_size[0], is_iso[0], nvals[0], ffi.NULL
                ))

        elif by_col and is_full:
            _check(lib.GxB_Matrix_pack_FullC(
                _A[0],
                Ax, Ax_size[0], is_iso[0], ffi.NULL
                ))

        elif by_row and is_full:
            _check(lib.GxB_Matrix_pack_FullR(
                _A[0],
                Ax, Ax_size[0], is_iso[0], ffi.NULL
                ))
        else:
            raise TypeError('Unknown format {format[0]}')

        from . import Matrix
        A = Matrix(_A, atype)
        A.hyper_switch = hyper_switch[0]
        return A

