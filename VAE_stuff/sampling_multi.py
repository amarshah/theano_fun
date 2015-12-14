import argparse
import itertools
import cPickle
import theano
import numpy as np
from matplotlib import cm, pyplot
from blocks.bricks import Random
from blocks.graph import ComputationGraph
from blocks.select import Selector
#from blocks.serialization import load
#from blocks.extensions import Load

def make_sampling_computation_graph(model_path, num_samples):
    f = file(model_path, 'rb')
    model = cPickle.load(f)#main_loop = load(model_path)#
    f.close()
    #model = main_loop.model
    selector = Selector(model.top_bricks)
    decoder_mlp1, = selector.select('/decoder_network1').bricks
    decoder_mlp2, = selector.select('/decoder_network2').bricks
    decoder_mlp3, = selector.select('/decoder_network3').bricks
    theano_rng = Random().theano_rng

    z2 = theano_rng.normal(size=(num_samples, decoder_mlp1.input_dim),
                           dtype=theano.config.floatX)

    h2 = decoder_mlp1.apply(z2) 
    h2 = h2[:, :50] + theano.tensor.exp(0.5 * h2[:, 50:]) * theano_rng.normal(size=(num_samples, 50),
                                                                              dtype=theano.config.floatX)


    z1 = theano_rng.normal(size=(num_samples, 10),
                           dtype=theano.config.floatX)

    h1 = decoder_mlp2.apply(theano.tensor.concatenate([h2, z1], axis=1)) 
    h1 = h1[:, :50] + theano.tensor.exp(0.5 * h1[:, 50:]) * theano_rng.normal(size=(num_samples, 50),
                                                                              dtype=theano.config.floatX)

    p = decoder_mlp3.apply(theano.tensor.concatenate([h1, h2], axis=1)).reshape((num_samples, 28, 28))

    return ComputationGraph([p])


def main(computation_graph):
    f = theano.function(computation_graph.inputs, computation_graph.outputs)
    samples_list = f()
    for samples in samples_list:
        print samples.min()
        print samples.max()
        figure, axes = pyplot.subplots(nrows=nrows, ncols=ncols)
        for n, (i, j) in enumerate(itertools.product(xrange(nrows),
                                                     xrange(ncols))):
            ax = axes[i][j]
            ax.axis('off')
            ax.imshow(samples[n], cmap=cm.Greys_r, interpolation='nearest')
    pyplot.show()


if __name__ == "__main__":
    nrows, ncols = 10, 10
    num_examples = nrows * ncols

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    args = parser.parse_args()

    main(make_sampling_computation_graph(args.model_path, num_examples))
