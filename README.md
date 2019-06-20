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

    a[3]                      # extract the 3rd row as vector
    a[:,3]                    # extract the 3rd column as vector
    a[3,3]                    # extract element at row 3, col 3
    a[2:4,3:8]                # extract submatrix by row/col ranges
    a[(2,1,4):(3,5)]          # extract submatrix by row/col
    a[3:10:2]                 # extract vector range by steps
    a[10:1:-2]                # extract vector descending range by steps
    
    aT = a.transpose()        # transpose of a

    b = Matrix.from_type(lib.GrB_FP32, 10, 10) # 10x10 matrix of 32-bit floats

    a @ b                     # mxm(a, b) with default PLUS_TIMES semiring

    with min_plus:
        a @ b                 # mxm(a, b) with MIN_PLUS semiring

    m = Matrix.from_type(bool, 10, 10)  # a 10x10 boolean matrix (same as GrB_BOOL)

    with min_plus(mask=m, inp1='tran'):
        a @ b                 # min_plus masked mxm(a, b) with transposed b

    dupa = Matrix.dup(a)      # make a dup of a
    a.clear()                 # clear a
```
    

# Introduction

For a long time, mathematicians have known that matrices are powerful
representations of graphs, as described [in this mathmatical
introduction to
GraphBLAS](http://www.mit.edu/~kepner/GraphBLAS/GraphBLAS-Math-release.pdf)
by [Dr. Jermey Kepner](http://www.mit.edu/~kepner/) head and founder
of [MIT Lincoln Laboratory Supercomputing
Center](http://news.mit.edu/2016/lincoln-laboratory-establishes-supercomputing-center-0511).

As Kepner's paper describes, there are two useful matrix
representations of graphs: [Adjacency
Matrices](https://en.wikipedia.org/wiki/Adjacency_matrix) and
[Incidence Matrices](https://en.wikipedia.org/wiki/Incidence_matrix).
For this introduction we will focus on the adjacency type as they are
simpler, but the same ideas apply to both, and it is easy to switch
back and forth between them.

![Alt text](./docs/AdjacencyMatrix.svg)

(Image Credit: [Dr. Jermey Kepner](http://www.mit.edu/~kepner/))

On the left is a *directed* graph, and on the right, the adjacency
matrix that represents it. The matrix has a row and column for every
vertex.  If there is an going from node A to B, then there will be a
value present in the intersection of As row with Bs column.  For
example, vertex 1 connects to 4, so there is a value (dot) at the
intersction of the first row and the fourth column.  4 also connects
*back* to 1 so there are two values in the matrix to represent these
two edges, the one at the (1, 4) position and the other at the (4,1)
position.
