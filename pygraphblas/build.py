from cffi import FFI
ffibuilder = FFI()

ffibuilder.set_source(
    "_pygraphblas",
    r"""#include "LAGraph.h" 

    // missing from LAGraph.h for the moment
    GrB_Info LAGraph_Vector_to_dense
    (
        GrB_Vector *vdense,     // output vector
        GrB_Vector v,           // input vector
        void *id                // pointer to value to fill vdense with
     ) ;

    """,
    libraries=['graphblas', 'lagraph'])

gb_cdef = open('pygraphblas/cdef/gb_2.3.3_cdef.h')
#gb_cdef = open('pygraphblas/cdef/gb_3.0.0_cdef.h')
la_cdef = open('pygraphblas/cdef/la_cdef.h')

ffibuilder.cdef(gb_cdef.read())
ffibuilder.cdef(la_cdef.read())

if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
