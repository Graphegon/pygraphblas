# pygraphblas

SuitSparse:GraphBLAS for Python

# Summary

pygraphblas is a python extension that bridges [The GraphBLAS
API](http://graphblas.org) with the [Python](https://python.org)
programming language.

GraphBLAS is a sparse linear algebra API optimized for processing
graphs encoded as sparse matrices and vectors.  In addition to common
real/integer matrix algebra operations, GraphBLAS supports up to 960
different [Semiring](https://en.wikipedia.org/wiki/Semiring) algebra
operations, that can be used as basic building blocks to implement a
wide variety of graph algorithms. See
[Applications](https://en.wikipedia.org/wiki/Semiring#Applications)
from Wikipedia for some specific examples.

pygraphblas leverages the expertise in the field of sparse matrix
programming by [The GraphBLAS Forum](http://graphblas.org) and uses
the
[SuiteSparse:GraphBLAS](http://faculty.cse.tamu.edu/davis/GraphBLAS.html)
API implementation. SuiteSparse:GraphBLAS is brought to us by the work
of [Dr. Tim Davis](http://faculty.cse.tamu.edu/davis/welcome.html),
professor in the Department of Computer Science and Engineering at
Texas A&M University.  [News and
information](http://faculty.cse.tamu.edu/davis/news.html) can provide
you with a lot more background information, in addition to the
references below.

pygraphblas uses the excellent
[CFFI](https://cffi.readthedocs.io/en/latest/) library and is tested
with CPython 3.5+.  CPython 2 will not be supported.

# Intro

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
simpler, but the same ideas apply to both, both are suported by
GraphBLAS, and it is easy to switch back and forth between them.

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

One practical problem with matrix-encoding graphs is that most
real-world graphs tend to be sparse, as above, only 12 of 49 possible
elements have a value. Those that have values tend to be scattered
uniformally across the matrix (for "typical" graphs), so dense linear
algebra libraries like BLAS or numpy do not encode or operate on them
efficiently, as the relevant data is mostly empty memory with actual
data elements spaced far apart.  This wastes memory and cpu resources,
and defeats CPU caching mechanisms.

For example, suppose a fictional social network has 1 billion users,
and each user has about 100 friends, which means there are about 100
billion (1e+11) connections in the graph.  A dense matrix large enough
to hold this graph would need (1 billion)^2 or
(1,000,000,000,000,000,000), a "quintillion" elements, but only 1e+11
of them would have meaningful values, leaving only 0.0000001 of the
matrix being utilized.

By using a sparse matrix instead of dense, only the elements used are
actually stored in the matrix. The parts of the matrix with no value
are interpreted as an "algebraic zero" value, which might not be the
actual number zero, but other values like positive or negative
infinity depending on the particular semiring operations applied to
the matrix.  The math used with sparse matrices is exactly the same as
dense, the sparsity of the data doesn't matter to the math, but it
does matter to how efficiently the matrix is implemented internally.

pygraphblas is a python module that provides access to two new
high-level types: `Matrix` and `Vector`, as well as the low-level
GraphBLAS api to manipulate these types.  pygraphblas uses the amazing
[CFFI](https://cffi.readthedocs.io/en/latest/) library to wrap the low
level C-api while the high level types are normal Python classes with
a "MATLAB-like" interface.

# API

The pygraphblas package contains the following sub-modules:

    - `pygrablas.matrix` contains the Matrix type

    - `pygrablas.vector` contains the Vector type

    - `pygrablas.descriptor` contains descriptor types

    - `pygrablas.semiring` contains Semiring types

    - `pygrablas.binaryop` contains BinaryOp types

    - `pygrablas.base` contains low-level API and FFI wrappers.

Full API documentation coming soon, for now, check out the complete
tests for usage.

# Differences from scipy.sparse

`scipy.sparse` is an excellent module but has a number of differences
from pygraphblas.  Future work on pygraphblas will allow scipy.sparse
matrices to be used interchangeably with pygraphblas.

- pygraphblas wraps GraphBLAS, all concepts are derived from
  GraphBLAS.

- GraphBLAS abstracts different backends such as threaded, OpenMP,
  and CUDA.

- scipy.sparse does not provide semiring algebras, Graph
  algorithms using matrices without semirings is more tedious.

- scipy.sparse has no sparse Vector type.  You can of course
  simulate them with 1xN matricies.

# TODO

- GxB_select()

- GxB_get/set()

- All the docstrings.

- ReadTheDocs site.

- Jupyter Notebook tutorial.

- Construction from numpy.array and scipy.sparse
