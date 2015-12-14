

import theano, blocks, fuel
import theano.tensor as T
from theano.scalar.basic_scipy import gammaln
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

epsilon = 1.5#0.75
n_vis = 28 ** 2
n_hidden = 500
n_latent = 20
batch_size = 100
n_epochs = 4000
learning_rate = 0.00025
save_freq = 20
model_save = 'VAE_constrained4_model.pkl'

x = T.matrix('features')

# Define the graph

encoder_network = MLP(activations=[Tanh(), Tanh(), Tanh(), Identity()],
                      dims=[n_vis, n_hidden, n_hidden, n_hidden, n_latent],
                      biases_init=Constant(0),
                      name='encoder_network') 
encoder = encoder_network.apply(x)

z = encoder + epsilon * srng.uniform(size=encoder.shape, low=-1., high=1.)

decoder_network = MLP(activations=[Tanh(), Logistic()],
                      dims=[n_latent, n_hidden, n_vis],
                      biases_init=Constant(0),
                      name='decoder_network') 
decoder_p = decoder_network.apply(z)


# Define the cost
 
prior_term = -0.5 * (n_latent * T.log(2 * 3.14159) + (z ** 2).sum(axis=1))
reconstruction_term = (x * T.log(decoder_p) + 
                       (1 - x) * T.log(1 - decoder_p)).sum(axis=1) 
log_ball_volume = n_latent * (0.5 * T.log(3.14159) 
                              + T.log(epsilon)) - gammaln(0.5 * n_latent + 1) 
cost = -(prior_term + reconstruction_term).mean()
cost.name = 'negative_log_likelihood'


# Initialize the parameters

encoder_network._push_initialization_config()
for layer in encoder_network.linear_transformations:
    layer.weights_init = Uniform(
        width=12. / (layer.input_dim + layer.output_dim))
encoder_network.initialize()

decoder_network._push_initialization_config()
for layer in decoder_network.linear_transformations:
    layer.weights_init = Uniform(
        width=12. / (layer.input_dim + layer.output_dim))
decoder_network.initialize()


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
