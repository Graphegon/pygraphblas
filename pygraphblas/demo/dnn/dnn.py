from pygraphblas import FP32
from . import timing

@timing
def dnn(W, B, Y):
    for w, b in zip(W, B):    # for every weight, bias matrix
        Y.mxm(w, out=Y)       # Y = Y @ w
        with FP32.PLUS_PLUS:  # with PLUS_PLUS semiring:
            Y.mxm(b, out=Y)   # Y = Y @ B
        Y.select('>0', out=Y) # select all >0 from Y
        M = Y.select('>', 32) # select all > 32
        if len(M):            # if any > 32
            Y[M] = 32         # truncate to 32
    return Y

