from pygraphblas import FP32
from . import timing

@timing
def dnn(W, B, Y):
    for w, b in zip(W, B):
        Y = Y.mxm(w, out=Y)
        with FP32.PLUS_PLUS:
            Y = Y.mxm(b, out=Y)
        Y.select('>0', out=Y)
        M = Y.select('>', 32)
        if len(M):
            Y[M] = 32
    return Y

