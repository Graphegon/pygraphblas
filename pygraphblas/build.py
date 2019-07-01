from cffi import FFI
ffibuilder = FFI()

ffibuilder.set_source(
    "_pygraphblas",
    r"""#include "LAGraph.h" """,
    libraries=['graphblas', 'lagraph'])

gb_cdef = open('pygraphblas/cdef/gb_2.3.3_cdef.h')
la_cdef = open('pygraphblas/cdef/la_cdef.h')

ffibuilder.cdef(gb_cdef.read())
ffibuilder.cdef(la_cdef.read())

if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
