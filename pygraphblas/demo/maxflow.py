from pygraphblas import Matrix, Vector
from pygraphblas.semiring import max_times_fp64
from pygraphblas.binaryop import max_fp64, Accum

# The graph as a list of row and column indices and values
data = [
    [1, 1, 2, 4, 4, 5, 6],  # rows
    [2, 4, 3, 5, 6, 3, 5],  # cols
    [0.2, 0.8, 0.3, 0.6, 1.0, 0.7, 0.9],  # vals
]

# create the matrix from the data
m = Matrix.from_lists(*data)


def maxflow_direct(matrix, start):
    v = Vector.from_type(matrix.gb_type, matrix.nrows)
    v[start] = 0

    with max_times_fp64, Accum(max_fp64):
        for _ in range(matrix.nrows):
            w = Vector.dup(v)
            v @= matrix
            if w == v:
                break
        return v
