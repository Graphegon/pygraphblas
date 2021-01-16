from pygraphblas import FP32, binary_op
from . import timing
from pygraphblas.demo.gviz import draw_matrix
import imageio


@timing
def render_frame(prefix, Y, i):
    im = draw_matrix(Y, scale=2, labels=False)
    imageio.imwrite(prefix + str(i) + ".png", im)


@timing
def dnn(W, B, Y, movie=None):
    breakpoint()
    for i, (w, b) in enumerate(zip(W, B)):  # for every weight, bias matrix
        if movie is not None:
            render_frame(movie, w, i)
        Y = Y @ w
        with FP32.PLUS_PLUS:  # with PLUS_PLUS semiring:
            Y.mxm(b, out=Y)  # Y = Y @ B
        Y.select(">0", out=Y)  # select all >0 from Y
        M = Y.select(">", 32)  # select all > 32
        if len(M):  # if any > 32
            Y[M] = 32  # truncate to 32
    return Y


class ReLUNeuron(FP32):
    @binary_op(FP32)
    def TIMES(x, y):
        result = min(x + y, 32)
        if result < 0:
            return 0
        return result


ReLUNeuron_monoid = ReLUNeuron.new_monoid(FP32.MAX, ReLUNeuron.one)
ReLUNeuron_semiring = ReLUNeuron.new_semiring(ReLUNeuron_monoid, ReLUNeuron.TIMES)


@timing
def hyperdnn(nlayers, W, B, Y):
    frames = []
    for i in range(nlayers):
        Y @= W
        with ReLUNeuron_semiring:
            Y @= B
        Y.select(">0", out=Y)
        print(Y.nvals)
    return Y
