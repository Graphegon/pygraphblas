from cffi import FFI
from pathlib import Path

def build_ffi():
    ffibuilder = FFI()

    source = r"""
        #include <math.h>
        #include <stdint.h> 
        """
    p = Path('pygraphblas/cdef/LAGraph')

    with open(p / 'LAGraph.h') as la:
        source += la.read()

    with open(p / 'LAGraph_internal.h') as lai:
        source += lai.read()

    for fname in p.glob('*.c'):
        with open(fname) as f:
            code = f.read()
            code = code.replace(
                '#include "LAGraph_internal.h"',
                '// #include "LAGraph_internal.h"')
            code += """
#undef LAGRAPH_FREE_ALL
#undef ARG
            """
            source += code

    ffibuilder.set_source(
        "_pygraphblas",
        source,
        libraries=['graphblas'])

    gb_cdef = open('pygraphblas/cdef/GraphBLAS-3.2.0.h')
    la_cdef = open('pygraphblas/cdef/la_a6fcf0_cdef.h')
    ex_cdef = open('pygraphblas/cdef/extra.h')

    ffibuilder.cdef(gb_cdef.read())
    ffibuilder.cdef(la_cdef.read())
    ffibuilder.cdef(ex_cdef.read())
    return ffibuilder

ffibuilder = build_ffi()

if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
