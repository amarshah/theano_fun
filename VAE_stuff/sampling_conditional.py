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
    decoder_mlp2, = selector.select('/decoder_network2').bricks
    decoder_mlp1, = selector.select('/decoder_network1').bricks
    upsample_mlp2, = selector.select('/upsample_network2').bricks
    upsample_mlp1, = selector.select('/upsample_network1').bricks
    theano_rng = Random().theano_rng

    z2 = theano_rng.normal(size=(num_samples, decoder_mlp2.input_dim),
                           dtype=theano.config.floatX)

    h2_params = decoder_mlp2.apply(z2)
    length = int(h2_params.eval().shape[1]/2)
    h2_mu = h2_params[:, :length]
    h2_lognu = h2_params[:, length:]
    h2 = h2_mu + theano.tensor.exp(0.5 * h2_lognu) * theano_rng.normal(size=h2_mu.shape,
                                                                       dtype=h2_mu.dtype)
    
    z1 = theano_rng.normal(size=(num_samples, decoder_mlp1.input_dim),
                           dtype=theano.config.floatX)

    h1_tilde_params = decoder_mlp1.apply(z1)
    length = int(h1_tilde_params.eval().shape[1]/2)
    h1_tilde_mu = h1_tilde_params[:, :length]
    h1_tilde_lognu = h1_tilde_params[:, length:]
    h1_tilde = h1_tilde_mu + theano.tensor.exp(0.5 * h1_tilde_lognu) * theano_rng.normal(size=h1_tilde_mu.shape,
                                                                                         dtype=h1_tilde_mu.dtype)


    import pdb; pdb.set_trace()
    h1 = upsample_mlp1.apply(h2) + h1_tilde
  
    p = upsample_mlp2.apply(h1).reshape((num_samples, 28, 28))

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
