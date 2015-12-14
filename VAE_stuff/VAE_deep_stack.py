#import argparse

import theano, blocks, fuel
import theano.tensor as T

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
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten


srng = RandomStreams(seed=234)

epsilon = 0.1
n_vis = 28 ** 2
n_hidden1 = 500
n_latent1 = 100
n_hidden2 = 70
n_latent2 = 40
n_hidden3 = 40
n_latent3 = 20
batch_size = 100
n_epochs = 4000
learning_rate = 0.0005
save_freq = 20
model_save = 'VAE_deep_stack.pkl'

x = T.matrix('features')

# Define the graph

encoder_network1 = MLP(activations=[Tanh(), Identity()],
                       dims=[n_vis, n_hidden1, n_latent1],
                       biases_init=Constant(0),
                       name='encoder_network1') 
encoder1 = encoder_network1.apply(x)

h1 = encoder1 + epsilon * srng.uniform(size=encoder1.shape,
                                       low=-1., high=1., dtype=encoder1.dtype)

encoder_network2 = MLP(activations=[Tanh(), Identity()],
                       dims=[n_latent1, n_hidden2, n_latent2],
                       biases_init=Constant(0),
                       name='encoder_network2') 
encoder2 = encoder_network2.apply(encoder1)

h2 = encoder2 + epsilon * srng.uniform(size=encoder2.shape,
                                       low=-1., high=1., dtype=encoder2.dtype)

encoder_network3 = MLP(activations=[Tanh(), Identity()],
                       dims=[n_latent2, n_hidden3, n_latent3],
                       biases_init=Constant(0),
                       name='encoder_network3') 
encoder3 = encoder_network3.apply(encoder2)

h3 = encoder3 + epsilon * srng.uniform(size=encoder3.shape,
                                       low=-1., high=1., dtype=encoder3.dtype)

###

decoder_network1 = MLP(activations=[Tanh(), Logistic()],
                       dims=[n_latent3, n_hidden3, 2 * n_latent2],
                       biases_init=Constant(0),
                       name='decoder_network1') 
decoder1_params = decoder_network1.apply(h3)
decoder1_mu = decoder1_params[:, :n_latent2]
decoder1_lognu = decoder1_params[:, n_latent2:]


decoder_network2 = MLP(activations=[Tanh(), Logistic()],
                       dims=[n_latent2, n_hidden2, 2 * n_latent1],
                       biases_init=Constant(0),
                       name='decoder_network2') 
decoder2_params = decoder_network2.apply(h2)
decoder2_mu = decoder2_params[:, :n_latent1]
decoder2_lognu = decoder2_params[:, n_latent1:]


#z = encoder_mu + T.exp(0.5 * encoder_lognu) * srng.normal(size=encoder_mu.shape,
#                                                          dtype=encoder_mu.dtype)

decoder_network3 = MLP(activations=[Tanh(), Logistic()],
                       dims=[n_latent1, n_hidden1, n_vis],
                       biases_init=Constant(0),
                       name='decoder_network3') 
decoder3_p = decoder_network3.apply(h1)


# Define the cost

prior_term = -0.5 * (T.log(2 * 3.14159) + h3 ** 2).sum(axis=1)
reconstruct_term_1 = -0.5 * (T.log(2 * 3.14159) + decoder1_lognu
                             + (h2 - decoder1_mu) ** 2 / T.exp(decoder1_lognu)).sum(axis=1) 
reconstruct_term_2 = -0.5 * (T.log(2 * 3.14159) + decoder2_lognu
                             + (h1 - decoder2_mu) ** 2 / T.exp(decoder2_lognu)).sum(axis=1) 
reconstruct_term_3 = (x * T.log(decoder3_p) + 
                      (1 - x) * T.log(1 - decoder3_p)).sum(axis=1) 
 
cost = -(prior_term + reconstruct_term_1 + reconstruct_term_2 + reconstruct_term_3).mean()
cost.name = 'negative_lower_bound'


# Initialize the parameters

encoder_network1._push_initialization_config()
for layer in encoder_network1.linear_transformations:
    layer.weights_init = Uniform(
        width=12. / (layer.input_dim + layer.output_dim))
encoder_network1.initialize()

encoder_network2._push_initialization_config()
for layer in encoder_network2.linear_transformations:
    layer.weights_init = Uniform(
        width=12. / (layer.input_dim + layer.output_dim))
encoder_network2.initialize()

encoder_network3._push_initialization_config()
for layer in encoder_network3.linear_transformations:
    layer.weights_init = Uniform(
        width=12. / (layer.input_dim + layer.output_dim))
encoder_network3.initialize()

decoder_network1._push_initialization_config()
for layer in decoder_network1.linear_transformations:
    layer.weights_init = Uniform(
        width=12. / (layer.input_dim + layer.output_dim))
decoder_network1.initialize()

decoder_network2._push_initialization_config()
for layer in decoder_network2.linear_transformations:
    layer.weights_init = Uniform(
        width=12. / (layer.input_dim + layer.output_dim))
decoder_network2.initialize()

decoder_network3._push_initialization_config()
for layer in decoder_network3.linear_transformations:
    layer.weights_init = Uniform(
        width=12. / (layer.input_dim + layer.output_dim))
decoder_network3.initialize()


mnist = BinarizedMNIST(("train",), sources=('features',))

train_data_stream = Flatten(DataStream.default_stream(
        mnist,
        iteration_scheme=SequentialScheme(mnist.num_examples, batch_size=batch_size)))

monitor_data_stream = Flatten(DataStream.default_stream(
        mnist,
        iteration_scheme=SequentialScheme(mnist.num_examples, batch_size=batch_size)))

cg = ComputationGraph(cost)
algorithm = GradientDescent(cost=cost, 
                            parameters=cg.parameters,
                            step_rule=Scale(learning_rate=learning_rate))


monitor = DataStreamMonitoring(variables=[cost],
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
