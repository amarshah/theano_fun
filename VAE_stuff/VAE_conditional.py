""" 2 layer autoencoder with VAE trained on each layer """

import theano, blocks, fuel, numpy
import theano.tensor as T

from numpy.linalg import inv
from theano.tensor.shared_randomstreams import RandomStreams
from blocks.bricks import Linear, Tanh, Logistic, MLP, Identity
from blocks.initialization import Uniform, Constant
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions import ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.graph import ComputationGraph
from blocks.extensions import FinishAfter, Printing
from fuel.datasets import BinarizedMNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Flatten


srng = RandomStreams(seed=234)

n_vis = 28 ** 2
n_latent1 = 400
n_latent2 = 80

n_hidden1 = 100
n_noise1 = 20

n_hidden2 = 40
n_noise2 = 20

batch_size = 100
n_epochs = 4000
learning_rate = 0.00000025#0.0005
save_freq = 20
model_save = 'VAE_conditional.pkl'

x = T.matrix('features')
rng = numpy.random.RandomState(1234)


# Set up simple autoencoder using linear link function

width = 2
W = numpy.asarray(rng.uniform(low=-width,
                              high=width,
                              size=(n_vis, n_latent1)
                  ),
                  dtype=theano.config.floatX
    )
down_matrix1 = theano.shared(value=W, 
                             name='down_matrix1'
               )

W2 = 5/3.2 * numpy.dot(numpy.transpose(W), 
                       inv(numpy.dot(W, numpy.transpose(W)) + 5 * numpy.identity(n_vis)))
up_matrix2 = theano.shared(value=W2, 
                           name='up_matrix2'
             )
 
width = 5
W3 = numpy.asarray(rng.uniform(low=-width,
                               high=width,
                               size=(n_latent1, n_latent2)
                   ),
                   dtype=theano.config.floatX
    )
down_matrix2 = theano.shared(value=W3, 
                             name='down_matrix2'
               )

W4 = 5 * numpy.dot(numpy.transpose(W3), 
                   inv(numpy.dot(W3, numpy.transpose(W3)) + 5 * numpy.identity(n_latent1)))
up_matrix1 = theano.shared(value=W4, 
                           name='up_matrix1'
             )


h1 = T.dot(x, down_matrix1)
h2 = T.dot(h1, down_matrix2)


# VAE 2
# for the bottom hidden unit

encoder_network2 = MLP(activations=[Tanh(), Identity()],
                       dims=[n_latent2, n_hidden2, 2 * n_noise2],
                       biases_init=Constant(0),
                       name='encoder_network2') 
z2_params = encoder_network2.apply(h2)
z2_mu = z2_params[:, :n_noise2]
z2_lognu = z2_params[:, n_noise2:]

z2 = z2_mu + T.exp(0.5 * z2_lognu) * srng.normal(size=z2_mu.shape,
                                                 dtype=z2_mu.dtype)

decoder_network2 = MLP(activations=[Tanh(), Identity()],
                       dims=[n_noise2, n_hidden2, 2 * n_latent2],
                       biases_init=Constant(0),
                       name='decoder_network2') 
h2_params = decoder_network2.apply(z2)
h2_mu = h2_params[:, :n_latent2]
h2_lognu = h2_params[:, n_latent2:]

h2_sample = h2_mu + T.exp(0.5 * h2_lognu) * srng.normal(size=h2_mu.shape,
                                                        dtype=h2_mu.dtype)

h1_up = T.dot(h2_sample, up_matrix1)

# VAE 1
# for top hidden unit given upsampled bottom hidden unit

encoder_network1 = MLP(activations=[Tanh(), Identity()],
                       dims=[n_latent1, n_hidden1, 2 * n_noise1],
                       biases_init=Constant(0),
                       name='encoder_network1') 
z1_params = encoder_network1.apply(h1 - h1_up)
z1_mu = z1_params[:, :n_noise1]
z1_lognu = z1_params[:, n_noise1:]

z1 = z1_mu + T.exp(0.5 * z1_lognu) * srng.normal(size=z1_mu.shape,
                                                 dtype=z1_mu.dtype)

decoder_network1 = MLP(activations=[Tanh(), Identity()],
                       dims=[n_noise1, n_hidden1, 2 * n_latent1],
                       biases_init=Constant(0),
                       name='decoder_network1') 
h1_params = decoder_network1.apply(z1)
h1_mu = h1_up + h1_params[:, :n_latent1]
h1_lognu = h1_params[:, n_latent1:]

h1_sample = h1_mu + T.exp(0.5 * h1_lognu) * srng.normal(size=h1_mu.shape,
                                                        dtype=h1_mu.dtype)



x_probs = T.nnet.sigmoid(T.dot(h1_sample, up_matrix2))


# Define the cost

KL_term2 = -0.5 * (1 + z2_lognu -T.exp(z2_lognu) - z2_mu ** 2).sum(axis=1)
reconstruct2 = -0.5 * (T.log(2 * 3.14159) + h2_lognu 
                       + (h2 - h2_mu) ** 2 / T.exp(h2_lognu)).sum(axis=1)
VAE2 = (KL_term2 - reconstruct2).mean()
VAE2.name = 'VAE2'

KL_term1 = -0.5 * (1 + z1_lognu -T.exp(z1_lognu) - z1_mu ** 2).sum(axis=1)
reconstruct1 = -0.5 * (T.log(2 * 3.14159) + h1_lognu 
                       + (h1 - h1_mu) ** 2 / T.exp(h1_lognu)).sum(axis=1)
VAE1 = (KL_term1 - reconstruct1).mean()
VAE1.name = 'VAE1'

final_reconstruct = ((x * T.log(x_probs) 
                      + (1 - x) * T.log(1 - x_probs)).sum(axis=1)).mean()
final_reconstruct.name = 'final_reconstruct_error'

cost = (VAE1 + VAE2 - final_reconstruct).mean()
cost.name = 'negative_bound'


# Initialize the parameters

networks = [encoder_network1, decoder_network1,
            encoder_network2, decoder_network2]
params = []


for network in networks:
    network._push_initialization_config()
    for layer in network.linear_transformations:
        layer.weights_init = Uniform(
            width=12. / (layer.input_dim + layer.output_dim))
        for param in layer.parameters:
            params.append(param)                   
    network.initialize()



mnist = BinarizedMNIST(("train",), sources=('features',))

train_data_stream = Flatten(
    DataStream.default_stream(
        dataset=mnist,
        iteration_scheme=ShuffledScheme(mnist.num_examples, batch_size=batch_size)))

monitor_data_stream = Flatten(
    DataStream.default_stream(
        mnist,
        iteration_scheme=ShuffledScheme(mnist.num_examples, batch_size=batch_size)))

        
algorithm = GradientDescent(cost=cost, 
                            parameters=params,
                            step_rule=Scale(learning_rate=learning_rate))


monitor = DataStreamMonitoring(variables=[cost, VAE1, VAE2, final_reconstruct],
                               data_stream=monitor_data_stream,
                               prefix="train")

model = Model(cost)

main_loop = MainLoop(model=model,
                     data_stream=train_data_stream, 
                     algorithm=algorithm,
                     extensions=[monitor, 
                                 FinishAfter(after_n_epochs=n_epochs), 
                                 Printing(),
                                 ProgressBar(),
                                 Checkpoint(model_save,
                                            every_n_epochs=save_freq,
                                            save_separately=['model', 'log'])
                     ]
            )

main_loop.run() 
