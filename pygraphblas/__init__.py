"""pygraphblas is a python extension that bridges [The GraphBLAS
API](http://graphblas.org) with the [Python](https://python.org)
programming language.  It uses the
[CFFI](https://cffi.readthedocs.io/en/latest/) library to wrap the low
level GraphBLAS API and provides high level `pygraphblas.Matrix` and
`pygraphblas.Vector` Python types that make GraphBLAS simple and easy.

See the [Github README](https://github.com/Graphegon/pygraphblas) for
details on how to install pygraphblas. Once installed, the library can
be imported for use:

>>> from pygraphblas import *

The core idea of the GraphBLAS is the mathematical duality between a
graph and a `pygraphblas.Matrix`.  As illustrated here, a graph can be
expressed as a `pygraphblas.Matrix` and vice versa.

`pygraphblas.Matrix` is the primary object of The GraphBLAS API.
There are many ways to contstruct them, but a simple approach is to
provide three lists of data, the first are are lists of the row and
column positions that define the begining and end of a graph edge, and
the third list is the weight for that edge:

>>> I = [0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6]
>>> J = [1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4]
>>> V = [True for _ in range(len(I))]
>>> M = Matrix.from_lists(I, J, V)
>>> print(M)
      0  1  2  3  4  5  6
  0|     t     t         |  0
  1|              t     t|  1
  2|                 t   |  2
  3|  t     t            |  3
  4|                 t   |  4
  5|        t            |  5
  6|        t  t  t      |  6
      0  1  2  3  4  5  6
>>> from pygraphblas.gviz import draw_graph
>>> g = draw_graph(M, show_weight=False, 
...     filename='/docs/imgs/Matrix_from_lists2')

![Matrix_from_lists2.png](../imgs/Matrix_from_lists2.png)

GraphBLAS is a sparse [Linear
Algebra](https://en.wikipedia.org/wiki/Linear_algebra) API optimized
for processing graphs encoded as sparse matrices and vectors.  In
addition to common real/integer `pygraphblas.Matrix` algebra
operations, GraphBLAS supports over a thousand different
[Semiring](https://en.wikipedia.org/wiki/Semiring) algebra operations,
that can be used as basic building blocks to implement a wide variety
of graph algorithms. See
[Applications](https://en.wikipedia.org/wiki/Semiring#Applications)
from Wikipedia for some specific examples.

The core operation of Linear Algebra is [Matrix
Multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication).
In this GraphBLAS duality, this is an operation along the edges of a
graph from nodes to their adjacenct neighbors, taking one step in a
[Breadth First
Search](https://en.wikipedia.org/wiki/Breadth-first_search) across the
graph:



pygraphblas leverages the expertise in the field of sparse matrix
programming by [The GraphBLAS Forum](http://graphblas.org) and uses
the
[SuiteSparse:GraphBLAS](http://faculty.cse.tamu.edu/davis/GraphBLAS.html)
API implementation. SuiteSparse:GraphBLAS is brought to us by the work
of [Dr. Tim Davis](http://faculty.cse.tamu.edu/davis/welcome.html),
professor in the Department of Computer Science and Engineering at
Texas A&M University.  [News and
information](http://faculty.cse.tamu.edu/davis/news.html) can provide
you with a lot more background information.

"""

__pdoc__ = {
    "base": False,
    "build": False,
}

from .base import (
    lib,
    ffi,
    GxB_INDEX_MAX,
    GxB_IMPLEMENTATION,
    GxB_SPEC,
    options_get,
    options_set,
)

lib.LAGraph_init()

from .semiring import build_semirings
from .binaryop import build_binaryops, Accum, binary_op
from .unaryop import build_unaryops
from .monoid import build_monoids
from .matrix import Matrix
from .vector import Vector
from .scalar import Scalar
from . import descriptor

__pdoc__ = {
    "base": False,
    "build": False,
    "unaryop": False,
    "binaryop": False,
    "monoid": False,
    "semiring": False,
    "matrix": False,
    "vector": False,
    "scalar": False,
    "types": False,
}

build_semirings(__pdoc__)
build_binaryops(__pdoc__)
build_unaryops(__pdoc__)
build_monoids(__pdoc__)

from .types import (
    FP64,
    FP32,
    FC64,
    FC32,
    INT64,
    INT32,
    INT16,
    INT8,
    UINT64,
    UINT32,
    UINT16,
    UINT8,
    BOOL,
)

__all__ = [
    "GxB_INDEX_MAX",
    "GxB_IMPLEMENTATION",
    "GxB_SPEC",
    "options_set",
    "options_get",
    "Matrix",
    "Vector",
    "Scalar",
    "FP64",
    "FP32",
    "FC64",
    "FC32",
    "INT64",
    "INT32",
    "INT16",
    "INT8",
    "UINT64",
    "UINT32",
    "UINT16",
    "UINT8",
    "BOOL",
    "descriptor",
    "Accum",
    "binary_op",
]

GxB_INDEX_MAX = GxB_INDEX_MAX
"""Maximum key size for SuiteSparse, defaults to `2**60`."""

GxB_IMPLEMENTATION = GxB_IMPLEMENTATION
""" Tuple containing GxB_IMPLEMENTATION (MAJOR, MINOR, SUB) """

GxB_SPEC = GxB_SPEC
""" Tuple containing GxB_SPEC (MAJOR, MINOR, SUB) """


def run_doctests():   # pragma: no cover
    import sys, doctest

    this = sys.modules[__name__]
    for mod in (this, matrix, descriptor, base):
        doctest.testmod(mod, optionflags=doctest.ELLIPSIS)
