"""pygraphblas is a python extension that bridges [The GraphBLAS
API](http://graphblas.org) with the [Python](https://python.org)
programming language.  It uses the
[CFFI](https://cffi.readthedocs.io/en/latest/) library to wrap the low
level GraphBLAS API and provides high level Matrix and Vector Python
types that make GraphBLAS simple and easy.

The core idea of the GraphBLAS is the mathematical duality between a
graph and a matrix.  As illustrated here, a graph can be expressed as
a matrix and vice versa.

![Adjacency Matrix](../AdjacencyMatrix.png)

GraphBLAS is a sparse linear algebra API optimized for processing
graphs encoded as sparse matrices and vectors.  In addition to common
real/integer matrix algebra operations, GraphBLAS supports over a
thousand different [Semiring](https://en.wikipedia.org/wiki/Semiring)
algebra operations, that can be used as basic building blocks to
implement a wide variety of graph algorithms. See
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
you with a lot more background information.

"""
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

from .matrix import Matrix
from .vector import Vector
from .scalar import Scalar
from .semiring import build_semirings
from .binaryop import build_binaryops
from .unaryop import build_unaryops
from .monoid import build_monoids

build_semirings()
build_binaryops()
build_unaryops()
build_monoids()

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

from . import semiring
from . import binaryop
from . import unaryop
from . import monoid
from . import descriptor

__all__ = [
    "lib",
    "ffi",
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
    "monoid",
    "unaryop",
    "binaryop",
    "semiring",
    "types",
    "descriptor",
]

GxB_INDEX_MAX = GxB_INDEX_MAX
"""Maximum key size for SuiteSparse, defaults to `2**60`."""

GxB_IMPLEMENTATION = GxB_IMPLEMENTATION
""" Tuple containing GxB_IMPLEMENTATION (MAJOR, MINOR, SUB) """

GxB_SPEC = GxB_SPEC
""" Tuple containing GxB_SPEC (MAJOR, MINOR, SUB) """

__pdoc__ = {
    "base": False,
    "build": False,
    }
