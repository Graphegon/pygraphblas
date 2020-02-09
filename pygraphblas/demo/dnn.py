import os
from functools import wraps, partial
from time import time
from statistics import mean
from pathlib import Path
from pygraphblas import *
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count

NFEATURES = 60000
BIAS = {1024: -0.3, 4096: -0.35, 16384: -0.4, 65536: -0.45}

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f' % (f.__name__, te-ts))
        return result
    return wrap

@timing
def dnn(W, B, Y):
    for w, b in zip(W, B):
        Y = Y @ w
        with plus_plus:
            Y = Y @ b
        Y = Y.select('>0')
        M = Y.select('>', 32)
        if len(M):
            Y[M] = 32
    return Y

@timing
def load_images(neurons, dest):
    fname = '{}/sparse-images-{}.{}'
    binfile = fname.format(dest, neurons, 'ssb')
    if Path(binfile).exists():
        return Matrix.from_binfile(binfile.encode('ascii'))
    images = Path(fname.format(dest, neurons, 'tsv'))
    with images.open() as i:
        m = Matrix.from_tsv(i, FP32, NFEATURES, neurons)
        m.to_binfile(binfile.encode('ascii'))
        return m

def load_categories(neurons, nlayers, dest):
    fname = '{}/neuron{}-l{}-categories.tsv'
    cats = Path(fname.format(dest, neurons, nlayers))
    result = Vector.from_type(BOOL, NFEATURES)
    with cats.open() as i:
        for line in i.readlines():
            result[int(line.strip())-1] = True
    return result

def load_layer(i, dest):
    fname = '{}/neuron{}/n{}-l{}.{}'
    binfile = fname.format(dest, neurons, neurons, str(i+1), 'ssb')
    if Path(binfile).exists():
        return Matrix.from_binfile(binfile.encode('ascii'))
    l = Path(fname.format(dest, neurons, neurons, str(i+1), 'tsv'))
    with l.open() as f:
        m = Matrix.from_tsv(f, FP32, neurons, neurons)
        m.to_binfile(binfile.encode('ascii'))
        return m

@timing
def generate_layers(neurons, nlayers, dest):
    neurons = Path('{}/neuron{}'.format(dest, neurons))
    with ThreadPool(cpu_count()) as pool:
        return pool.map(partial(load_layer, dest=dest), range(nlayers))

@timing
def generate_bias(neurons, nlayers):
    result = []
    for i in range(nlayers):
        bias = Matrix.from_type(FP32, neurons, neurons)
        for i in range(neurons):
            bias[i,i] = BIAS[neurons]
        bias.nvals # causes async completion
        result.append(bias)
    return result

@timing
def run(neurons, images, layers, bias, dest):
    result = dnn(layers,
                 bias,
                 images)
    r = result.reduce_vector()
    cats = r.apply(lib.GxB_ONE_BOOL, out=Vector.from_type(BOOL, r.size))
    truecats = load_categories(neurons, nlayers, dest)
    assert cats == truecats


num_neurons = [1024, 4096, 16384, 65536]
num_layers = [120, 480, 1920]

if __name__ == '__main__':
    dest = os.getenv('DEST')
    neurons = os.getenv('NEURONS')
    nlayers = os.getenv('NLAYERS')

    if neurons and nlayers:
        neurons = int(neurons)
        nlayers = int(nlayers)
        images = load_images(neurons, dest)
        layers = generate_layers(neurons, nlayers, dest)
        bias = generate_bias(neurons, nlayers)
        run(neurons, images, layers, bias, dest)
    else:
        for neurons in num_neurons:
            print('Building layers for %s neurons' % neurons)
            layers = generate_layers(neurons, 1920, dest)
            bias = generate_bias(neurons, 1920)
            images = load_images(neurons, dest)
            for nlayers in num_layers:
                print('Benching %s neurons %s layers' % (neurons, nlayers))
                run(neurons, images, layers[:nlayers], bias[:nlayers], dest)
