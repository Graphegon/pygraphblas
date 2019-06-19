from cffi import FFI
ffibuilder = FFI()

ffibuilder.set_source(
    "_pygraphblas",
    r"""#include "GraphBLAS.h" """,
    libraries=['graphblas'])

gb_cdef = open('pygraphblas/gb_cdef.h')

ffibuilder.cdef(gb_cdef.read())

if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
