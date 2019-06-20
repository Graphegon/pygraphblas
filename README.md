# pygraphblas

GraphBLAS for Python

# Summary

pygraphblas is a python extension that bridges
[The GraphBLAS API](http://graphblas.org) with the
[Python](https://python.org) programming language.

GraphBLAS is a sparse linear algebra API optimized for processing
graphs encoded as sparse matrices and vectors.  In addition to common
real/integer matrix algebra operations, GraphBLAS supports up to 960
different "semiring" algebra operations, that can be used as basic
building blocks to implement a wide variety of graph algorithms.

pygraphblas leverages the expertise in the field of sparse matrix
programming by [The GraphBLAS Forum](http://graphblas.org) and uses
the
[SuiteSparse:GraphBLAS](http://faculty.cse.tamu.edu/davis/GraphBLAS.html)
API implementation. SuiteSparse:GraphBLAS is brought to us by the work
of [Dr. Tim Davis](http://faculty.cse.tamu.edu/davis/welcome.html),
professor in the Department of Computer Science and Engineering at
Texas A&M University.
[News and information](http://faculty.cse.tamu.edu/davis/news.html)
can provide you with a lot more background information, in addition to
the references below.

# Examples

Some example usage of the library.  Not all of these work yet!

```
    from pygraphblas.matrix import Matrix
    from pygraphblas.semiring import min_plus
    from pygraphblas import lib

    a = Matrix.from_type(int, 10, 10)   # a 10x10 matrix of integers (GrB_INT64)

    a.read_mm('data_file.mm') # read MatrixMarket format

    a[3]                      # extract the 3rd row as vector
    a[:,3]                    # extract the 3rd column as vector
    a[3,3]                    # extract element at row 3, col 3
    a[2:4,3:8]                # extract submatrix by row/col ranges
    a[(2,1,4):(3,5)]          # extract submatrix by explicit row/col
    a[3:10:2]                 # extract vector range by steps
    a[10:1:-2]                # extract vector descending range by steps
    
    aT = a.transpose()        # transpose of a

    b = Matrix.from_type(lib.GrB_FP32, 10, 10) # 10x10 matrix of 32-bit floats

    b.from_mm('...literal MatrixMarket string data...')

    a @ b                     # mxm(a, b) with default PLUS_TIMES semiring

    with min_plus:
        c = a @ b             # mxm(a, b) with MIN_PLUS semiring

    import numpy as np
    m = Matrix.from_numpy(np.random.choice(a=[True, False], size=(10, 10))) # from numpy array

    with min_plus(mask=m, inp1='tran'):
        c = a @ b             # min_plus masked mxm(a, b) with transposed b

    dupa = Matrix.dup(a)      # make a dup of a
    a.clear()                 # clear a

```
    
