import os
from .dnn import *
from .radix import *
from .challenge import *

num_neurons = [1024, 4096, 16384, 65536]
num_layers = [120, 480, 1920]

dest = os.getenv("DEST")
neurons = os.getenv("NEURONS")
nlayers = os.getenv("NLAYERS")

if neurons and nlayers:
    neurons = int(neurons)
    nlayers = int(nlayers)
    images = load_images(neurons, dest)
    layers = load_layers(neurons, dest, nlayers)
    bias = generate_bias(neurons, nlayers)
    run(neurons, images, layers, bias, dest)
else:
    for neurons in num_neurons:
        print("Building layers for %s neurons" % neurons)
        layers = load_layers(neurons, 1920, dest)
        bias = generate_bias(neurons, 1920)
        images = load_images(neurons, dest)
        for nlayers in num_layers:
            print("Benching %s neurons %s layers" % (neurons, nlayers))
            run(neurons, images, layers[:nlayers], bias[:nlayers], dest)
