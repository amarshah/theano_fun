import argparse
import itertools
import cPickle
import theano
from matplotlib import cm, pyplot
from blocks.bricks import Random
from blocks.graph import ComputationGraph
from blocks.select import Selector
from blocks.serialization import load


def make_sampling_computation_graph(model_path, num_samples):
  #  f = file(model_path, 'rb')
    main_loop = load(model_path)#cPickle.load(f)
  #  f.close()
    model = main_loop.model
    selector = Selector(model.top_bricks)
    decoder_mlp, = selector.select('/decoder_network').bricks
    theano_rng = Random().theano_rng

    z = theano_rng.normal(size=(num_samples, decoder_mlp.input_dim),
                          dtype=theano.config.floatX)
    p = decoder_mlp.apply(z).reshape((num_samples, 28, 28))

    return ComputationGraph([p])


def main(computation_graph):
    f = theano.function(computation_graph.inputs, computation_graph.outputs)
    samples_list = f()
    for samples in samples_list:
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
