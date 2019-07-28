
from pathlib import Path
from pygraphblas import Matrix, lib
from pygraphblas.semiring import plus_times_fp32, plus_plus_fp32

nfeatures = 60000
nneurons = 1024

def dnn(W, Bias, Y0):
    Y = Matrix.from_type(Y0.gb_type, nfeatures, nneurons)
    for layer, (w, b) in enumerate(zip(W, Bias)):
        with plus_times_fp32:
            print('Y @= w')
            (Y0 if layer == 0 else Y).mxm(w, out=Y)
        with plus_plus_fp32:
            print('Y @= b')
            Y.mxm(b, out=Y)
        Y.select(lib.GxB_GT_ZERO, out=Y)
        Y.apply(lib.LAGraph_YMAX_FP32, out=Y)
        print(repr(Y))
    return Y

def load_images():
    images = Path('sparse-images-1024.tsv')
    with images.open() as i:
        print('loading images.')
        Y0 = Matrix.from_tsv(i, lib.GrB_FP32, nfeatures, nneurons)
    return Y0

def generate_layers(layers=120):
    neurons = Path('./neuron1024')
    for i in range(layers):
        l = Path('./neuron1024/n1024-l{}.tsv'.format(str(i+1)))
        with l.open() as f:
            print('loading layer: ', i+1)
            yield Matrix.from_tsv(f, lib.GrB_FP32, nneurons, nneurons)

def generate_bias(layers=120):
    for i in range(layers):
        bias = Matrix.from_type(lib.GrB_FP32, nneurons, nneurons)
        for i in range(nneurons):
            bias[i,i] = 0.3
        bias.nvals # causes async completion
        yield bias

Y0 = load_images()
result = dnn(generate_layers(), generate_bias(), Y0)
with open('dnn_output.mm', 'w') as f:
    result.to_mm(f)

