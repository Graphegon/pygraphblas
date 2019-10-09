import gzip
from functools import wraps
from time import time
from pathlib import Path
from pygraphblas import Matrix, lib
from pygraphblas.semiring import plus_times_fp32, plus_plus_fp32
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count


nfeatures = 60000
nneurons = 1024

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
    Y = Matrix.from_type(Y0.gb_type, nfeatures, nneurons)
    for layer, (w, b) in enumerate(zip(W, Bias)):
        with plus_times_fp32:
            (Y0 if layer == 0 else Y).mxm(w, out=Y)
        with plus_plus_fp32:
            Y.mxm(b, out=Y)
        Y.select(lib.GxB_GT_ZERO, out=Y) # Y = Y > 0
        M = Y > 32
        if len(M):
            Y[M] = 32
    return Y

@timing
def dnn2(W, Bias, Y):
    Y = Matrix.from_type(Y0.gb_type, nfeatures, nneurons)
    for layer, (w, b) in enumerate(zip(W, Bias)):
        Y = Y.mxm((Y @ w), b, semiring=plus_plus_fp32)
        Y = Y > 0
        M = Y > 32
        if len(M):
            Y[M] = 32
    return Y

@timing
def load_images():
    images = Path('./dnn_demo/sparse-images-1024.tsv')
    with images.open() as i:
        return Matrix.from_tsv(i, lib.GrB_FP32, nfeatures, nneurons)

@timing
def load_categories():
    cats = Path('./dnn_demo/neuron1024-l120-categories.tsv')
    result = Vector.from_type(bool, nfeatures)
    with cats.open() as i:
        for line in i.readlines():
            result[int(line.strip())] = True
    return result

def load_layer(i):
    l = Path('./dnn_demo/neuron1024/n1024-l{}.tsv'.format(str(i+1)))
    with l.open() as f:
        return Matrix.from_tsv(f, lib.GrB_FP32, nneurons, nneurons)

@timing
def generate_layers(layers=120):
    result = []
    neurons = Path('./dnn_demo/neuron1024')
    with ThreadPool(cpu_count()) as pool:
        return pool.map(load_layer, range(layers))

@timing
def generate_bias(layers=120):
    result = []
    for i in range(layers):
        bias = Matrix.from_type(lib.GrB_FP32, nneurons, nneurons)
        for i in range(nneurons):
            bias[i,i] = 0.3
        bias.nvals # causes async completion
        result.append(bias)
    return result


if __name__ == '__main__':
    result = dnn(generate_layers(), generate_bias(), load_images())
    cat = load_categories()
