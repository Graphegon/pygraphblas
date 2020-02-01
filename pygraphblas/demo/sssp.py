from pygraphblas import Matrix, Vector
from pygraphblas.semiring import min_plus_int64
from pygraphblas.binaryop import min_int64, Accum

# The graph as a list of row and column indices and values
data = [
    [1,1,2,4,4,5,6],  # rows
    [2,4,3,5,6,3,5],  # cols
    [9,3,8,6,1,4,2],  # vals
]

# create the matrix from the data
m = Matrix.from_lists(*data)

def sssp(m, start):
    v = Vector.from_type(            # create a vector 
        m.type,                      # same type as m
        m.nrows                      # same size as rows of m
    )
    v[start] = 0                     # set the starting vertext distance
    
    for _ in range(m.nrows):         # for every row in m:
        w = Vector.dup(v)            # dup the vector
        v.vxm(                       # multiply vector by matrix 
            m, out=v,
            semiring=min_plus_int64, # with min_plus, 
            accum=min_int64          # acccumulate the minimum
        )
        if w == v:                   # if the result hasn't changed,
            break                    # exit early
    return v

# this is exact same as above, but using `with` syntax to specify
# semring and accumulator so that the `@` matmul syntax is used
# instead of explict vxm.

def sssp(matrix, start):
    v = Vector.from_type(matrix.type, matrix.nrows)
    v[start] = 0

    with min_plus_int64, Accum(min_int64):
        for _ in range(matrix.nrows):
            w = Vector.dup(v)
            v @= matrix
            if w == v:
                break
        return v
