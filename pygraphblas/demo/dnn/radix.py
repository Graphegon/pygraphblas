from pygraphblas import *
from operator import mul, eq, attrgetter
from functools import reduce
from . import timing


def permutation_matrix(size):
    P = Matrix.sparse(FP32, size, size)
    P[size - 1, 0] = 1.0
    for i in range(size - 1):
        P[i, i + 1] = 1.0
    return P


def mixed_topo_radix(topos):
    sizes = [reduce(mul, x) for x in topos]
    assert reduce(eq, sizes)
    size = sizes[0]
    layers = []
    P = permutation_matrix(size)

    for t in topos:
        place_value = 1
        for n in t:
            layer = Matrix.sparse(FP32, size, size)
            for j in range(n):
                layer += P ** (j * place_value)
            place_value *= n
            layers.append(layer)
    return layers


def ddnn(spec):
    return [Matrix.dense(FP32, spec[i], spec[i + 1]) for i in range(len(spec) - 1)]


def radixnet(topos, spec):
    return [d.kronecker(w) for d, w in zip(mixed_topo_radix(topos), ddnn(spec))]


def randomize(layers, damp=0.1):
    return [
        l.emult(Matrix.random(FP32, 12, 12, 1000), FP32.PLUS).apply_second(
            FP32.TIMES, damp
        )
        for l in layers
    ]

_rowgetter = attrgetter('nrows')

@timing
def hypergraph(mt, size=None):
    if size is None:
        size = sum(map(_rowgetter, mt)) + mt[-1].nrows
    r = Matrix.sparse(FP32, size, size)
    ioffset = 0
    joffset = 0
    for m in mt:
        joffset += m.nrows
        for c, (i, j, k) in enumerate(m):
            r[i + ioffset, j + joffset] = k
        ioffset += m.nrows
    return r
