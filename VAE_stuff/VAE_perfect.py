# VAE with perfect reconstruction


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

n_vis = 28 ** 2
block_dim = 7 * 16
n_block = n_vis / block_dim

batch_size = 100
n_epochs = 4000
learning_rate = 0.00075
save_freq = 20
model_save = 'VAE_perfect.pkl'

x = T.matrix('features')

# Define the graph

for block in xrange(n_block):
    


encoder_network1 = MLP(activations=[Tanh(), Identity()],
                       dims=[n_vis, n_hidden1, 2 * n_latent1],
                       biases_init=Constant(0),
                       name='encoder_network1') 
h1_params = encoder_network1.apply(x)
h1_mu = h1_params[:, :n_latent1]
h1_lognu = h1_params[:, n_latent1:]

h1_encode = h1_mu + T.exp(0.5 * h1_lognu) * srng.normal(size=h1_mu.shape,
                                                        dtype=h1_mu.dtype)

encoder_network2 = MLP(activations=[Tanh(), Identity()],
                       dims=[n_vis + n_latent1, n_hidden2, 2 * n_latent2],
                       biases_init=Constant(0),
                       name='encoder_network2')
h2_params = encoder_network2.apply(T.concatenate([x, h1_encode], axis=1))
h2_mu = h2_params[:, :n_latent2]
h2_lognu = h2_params[:, n_latent2:]

h2_encode = h2_mu + T.exp(0.5 * h2_lognu) * srng.normal(size=h2_mu.shape,
                                                        dtype=h2_mu.dtype)

encoder_network3 = MLP(activations=[Tanh(), Identity()],
                       dims=[n_latent1, n_hidden3, 2 * n_noise1],
                       biases_init=Constant(0),
                       name='encoder_network3') 
z1_params = encoder_network3.apply(h1_encode)
z1_mu = z1_params[:, :n_noise1]
z1_lognu = z1_params[:, n_noise1:]

z1 = z1_mu + T.exp(0.5 * z1_lognu) * srng.normal(size=z1_mu.shape,
                                                 dtype=z1_mu.dtype)


encoder_network4 = MLP(activations=[Tanh(), Identity()],
                       dims=[n_latent2, n_hidden4, 2 * n_noise2],
                       biases_init=Constant(0),
                       name='encoder_network4') 
z2_params = encoder_network4.apply(h2_encode)
z2_mu = z2_params[:, :n_noise2]
z2_lognu = z2_params[:, n_noise2:]

z2 = z2_mu + T.exp(0.5 * z2_lognu) * srng.normal(size=z2_mu.shape,
                                                 dtype=z2_mu.dtype)

#########

decoder_network1 = MLP(activations=[Tanh(), Identity()],
                       dims=[n_noise2, n_hidden4, 2 * n_latent2],
                       biases_init=Constant(0),
                       name='decoder_network1') 
h2_params = decoder_network1.apply(z2)
h2_mu = h2_params[:, :n_latent2]
h2_lognu = h2_params[:, n_latent2:]

h2_decode = h2_mu + T.exp(0.5 * h2_lognu) * srng.normal(size=h2_mu.shape,
                                                        dtype=h2_mu.dtype)

decoder_network2 = MLP(activations=[Tanh(), Identity()],
                       dims=[n_latent2 + n_noise1, n_hidden2, 2 * n_latent1],
                       biases_init=Constant(0),
                       name='decoder_network2') 
h1_params = decoder_network2.apply(T.concatenate([h2_decode, z1], axis=1))
h1_mu = h1_params[:, :n_latent1]
h1_lognu = h1_params[:, n_latent1:]

h1_decode = h1_mu + T.exp(0.5 * h1_lognu) * srng.normal(size=h1_mu.shape,
                                                        dtype=h1_mu.dtype)

decoder_network3 = MLP(activations=[Tanh(), Logistic()],
                       dims=[n_latent1 + n_latent2, n_hidden1, n_vis],
                       biases_init=Constant(0),
                       name='decoder_network3') 
decoder_p = decoder_network3.apply(T.concatenate([h1_decode, h2_decode], axis=1))


# Define the cost

KL_term = -0.5 * ((1 + z1_lognu -T.exp(z1_lognu) - z1_mu ** 2).sum(axis=1) 
          + (1 + z2_lognu -T.exp(z2_lognu) - z2_mu ** 2).sum(axis=1))
reconstruction_term = (x * T.log(decoder_p) + 
                       (1 - x) * T.log(1 - decoder_p)).sum(axis=1) 
cost = (KL_term -reconstruction_term).mean()
cost.name = 'negative_log_likelihood'


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

encoder_network4._push_initialization_config()
for layer in encoder_network4.linear_transformations:
    layer.weights_init = Uniform(
        width=12. / (layer.input_dim + layer.output_dim))
encoder_network4.initialize()

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
