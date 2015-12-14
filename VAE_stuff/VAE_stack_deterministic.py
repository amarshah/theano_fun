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

epsilon = 0.15#0.25
n_vis = 28 ** 2
n_hidden1 = 500
n_latent1 = 100
n_hidden2 = 50
n_latent2 = 30
batch_size = 100
n_epochs = 4000
learning_rate = 0.0001#0.0005
save_freq = 20
model_save = 'VAE_stack_deterministic.pkl'

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

decoder_network1 = MLP(activations=[Tanh(), Logistic()],
                       dims=[n_latent2, n_hidden2, n_latent1],
                       biases_init=Constant(0),
                       name='decoder_network1') 
decoder_mu = decoder_network1.apply(h2)
decoder_lognu = 2 * T.log(0.2 / 2) #decoder_params[:, n_latent1:]

#z = encoder_mu + T.exp(0.5 * encoder_lognu) * srng.normal(size=encoder_mu.shape,
#                                                          dtype=encoder_mu.dtype)

decoder_network2 = MLP(activations=[Tanh(), Logistic()],
                       dims=[n_latent1, n_hidden1, n_vis],
                       biases_init=Constant(0),
                       name='decoder_network2') 
decoder_p = decoder_network2.apply(h1)


# Define the cost

prior_term = -0.5 * (T.log(2 * 3.14159) + h2 ** 2).sum(axis=1)
reconstruct_term_1 = -0.5 * (T.log(2 * 3.14159) + decoder_lognu
                             + (h1 - decoder_mu) ** 2 / T.exp(decoder_lognu)).sum(axis=1) 
reconstruct_term_2 = (x * T.log(decoder_p) + 
                      (1 - x) * T.log(1 - decoder_p)).sum(axis=1) 
 
cost = -(prior_term + reconstruct_term_1 + reconstruct_term_2).mean()
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
