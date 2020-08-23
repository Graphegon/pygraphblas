from pygraphblas import *
from operator import mul, eq
from functools import reduce


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
