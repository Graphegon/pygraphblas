import os
import gzip
from functools import wraps
from time import time
from pathlib import Path
from pygraphblas import Matrix, Vector, lib
from pygraphblas.semiring import plus_times_fp32, plus_plus_fp32
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count


nfeatures = 60000

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        print('timing %r ...' % f.__name__)
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % (f.__name__, te-ts))
        return result
    return wrap

@timing
def dnn(W, Bias, Y0):
    Y = Y0
    for (w, b) in zip(W, Bias):
        Y = Y @ w
        with plus_plus_fp32:
            Y = Y @ b
        Y = Y > 0
        M = Y > 32
        if M:
            Y[M] = 32
    return Y

@timing
def load_images(nneurons):
    images = Path('./dnn_demo/sparse-images-{}.tsv'.format(nneurons))
    with images.open() as i:
        return Matrix.from_tsv(i, lib.GrB_FP32, nfeatures, nneurons)

@timing
def load_categories(nneurons, nlayers):
    cats = Path('./dnn_demo/neuron{}-l{}-categories.tsv'.format(nneurons, nlayers))
    result = Vector.from_type(bool, nfeatures)
    with cats.open() as i:
        for line in i.readlines():
            result[int(line.strip())-1] = True
    return result

def load_layer(i):
    l = Path('./dnn_demo/neuron{}/n{}-l{}.tsv'.format(nneurons, nneurons, str(i+1)))
    with l.open() as f:
        return Matrix.from_tsv(f, lib.GrB_FP32, nneurons, nneurons)

@timing
def generate_layers(nneurons, nlayers):
    result = []
    neurons = Path('./dnn_demo/neuron{}'.format(nneurons))
    with ThreadPool(cpu_count()) as pool:
        return pool.map(load_layer, range(nlayers))

@timing
def generate_bias(nneurons, nlayers):
    result = []
    for i in range(nlayers):
        bias = Matrix.from_type(lib.GrB_FP32, nneurons, nneurons)
        for i in range(nneurons):
            bias[i,i] = -0.3
        bias.nvals # causes async completion
        result.append(bias)
    return result


if __name__ == '__main__':
    nneurons = int(os.getenv('NNEURONS'))
    nlayers = int(os.getenv('NLAYERS'))

    result = dnn(generate_layers(nneurons, nlayers),
                 generate_bias(nneurons, nlayers),
                 load_images(nneurons))
    r = result.reduce_vector()
    cats = r.apply(lib.GxB_ONE_BOOL, out=Vector.from_type(bool, r.size))
    truecats = load_categories(nneurons, nlayers)
