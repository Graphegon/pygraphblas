from cffi import FFI
from pathlib import Path

def build_ffi():
    ffibuilder = FFI()

    source = r"""
        #include <math.h>
        #include <stdint.h>
        """
    p = Path('pygraphblas/cdef')
    l = Path('pygraphblas/cdef/LAGraph')

    with open(l / 'LAGraph.h') as lag:
        source += lag.read()

    with open(l / 'LAGraph_internal.h') as lai:
        source += lai.read()

    for fname in l.glob('*.c'):
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

    with open(p / 'usercomplex.c') as cmx:
        source += cmx.read()

    ffibuilder.set_source(
        "_pygraphblas",
        source,
        libraries=['graphblas'],
        extra_compile_args=['-std=c11', '-lm', '-Wno-pragmas', '-fopenmp'])

    with open(p / 'GraphBLAS-3.3.3.h') as gb_cdef:
        ffibuilder.cdef(gb_cdef.read())

    with open(p / 'la_a6fcf0_cdef.h') as la_cdef:
        ffibuilder.cdef(la_cdef.read())

    with open(p / 'extra.h') as ex_cdef:
        ffibuilder.cdef(ex_cdef.read())

    with open(p / 'usercomplex.h') as gb_cdef:
        ffibuilder.cdef(gb_cdef.read())

    return ffibuilder

ffibuilder = build_ffi()

if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
