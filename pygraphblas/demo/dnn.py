import os
from functools import wraps, partial
from time import time
from pathlib import Path
from pygraphblas import Matrix, Vector, lib
from pygraphblas.semiring import plus_times_fp32, plus_plus_fp32
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count


NFEATURES = 60000
BIAS = {1024: -0.3, 4096: -0.35, 16384: -0.4, 65536: -0.45} 

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
def load_images(nneurons, dest):
    images = Path('{}/sparse-images-{}.tsv'.format(dest, nneurons))
    with images.open() as i:
        return Matrix.from_tsv(i, lib.GrB_FP32, NFEATURES, nneurons)

@timing
def load_categories(nneurons, nlayers, dest):
    cats = Path('{}/neuron{}-l{}-categories.tsv'.format(dest, nneurons, nlayers))
    result = Vector.from_type(bool, NFEATURES)
    with cats.open() as i:
        for line in i.readlines():
            result[int(line.strip())-1] = True
    return result

def load_layer(i, dest):
    l = Path('{}/neuron{}/n{}-l{}.tsv'.format(dest, nneurons, nneurons, str(i+1)))
    with l.open() as f:
        return Matrix.from_tsv(f, lib.GrB_FP32, nneurons, nneurons)

@timing
def generate_layers(nneurons, nlayers, dest):
    neurons = Path('{}/neuron{}'.format(dest, nneurons))
    with ThreadPool(cpu_count()) as pool:
        return pool.map(partial(load_layer, dest=dest), range(nlayers), (cpu_count()/2))

@timing
def generate_bias(nneurons, nlayers):
    result = []
    for i in range(nlayers):
        bias = Matrix.from_type(lib.GrB_FP32, nneurons, nneurons)
        for i in range(nneurons):
            bias[i,i] = BIAS[nneurons]
        bias.nvals # causes async completion
        result.append(bias)
    return result


if __name__ == '__main__':
    nneurons = int(os.getenv('NNEURONS'))
    nlayers = int(os.getenv('NLAYERS'))
    dest = os.getenv('DEST')

    images = load_images(nneurons, dest)
    result = dnn(generate_layers(nneurons, nlayers, dest),
                 generate_bias(nneurons, nlayers),
                 images)
    r = result.reduce_vector()
    cats = r.apply(lib.GxB_ONE_BOOL, out=Vector.from_type(bool, r.size))
    truecats = load_categories(nneurons, nlayers, dest)
