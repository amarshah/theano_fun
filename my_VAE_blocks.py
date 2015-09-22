
import theano, blocks, fuel
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams
from blocks.bricks import Linear, Tanh, Logistic, MLP
from blocks.initialization import Uniform, Constant
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.main_loop import MainLoop
from blocks.graph import ComputationGraph
from blocks.extensions import FinishAfter, Printing
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten


srng = RandomStreams(seed=234)

n_vis = 28 ** 2
n_hidden = 500
n_latent = 20
batch_size = 100
n_epochs = 2
learning_rate = 0.001

x = T.matrix('x')

# Define the graph

encoder_layer_1 = Linear(name='encoder_layer_1', 
                       input_dim=n_vis, output_dim=n_hidden)
encoder_layer_1_out = Tanh().apply(encoder_layer_1.apply(x))

encoder_mu = Linear(name='encoder_mu',
                    input_dim=n_hidden, 
                    output_dim=n_latent).apply(encoder_layer_1_out)

encoder_lognu = Linear(name='encoder_lognu',
                       input_dim=n_hidden, 
                       output_dim=n_latent).apply(encoder_layer_1_out)

z = encoder_mu + T.exp(0.5 * encoder_lognu) * srng.normal(size=encoder_mu.shape,
                                                          dtype=encoder_mu.dtype)

decoder_p = MLP(activations=[Tanh(), Logistic()],
                dims=[n_latent, n_hidden, n_vis]).apply(z)


# Define the cost
 
KL_term = -0.5 * (1 + encoder_lognu -T.exp(encoder_lognu) - encoder_mu ** 2 ).sum(axis=1)
reconstruction_term = (x * T.log(decoder_p) + 
                       (1 - x) * T.log(1 - decoder_p)).sum(axis=1) 
cost = (KL_term -reconstruction_term).mean()
cost.name = 'negative_log_likelihood'


# Initialize the parameters

encoder_layer_1.weights_init = Uniform(width=12. / (n_vis + n_hidden))
encoder_layer_1.biases_init = Constant(0)
encoder_mu.weights_init = Uniform(width=12. / (n_hidden + n_latent))
encoder_mu.biases_init = Constant(0)
encoder_lognu.weights_init = Uniform(width=12. / (n_hidden + n_latent))
encoder_lognu.biases_init = Constant(0)


mnist = MNIST(("train",))
data_stream = Flatten(DataStream.default_stream(
        mnist,
        iteration_scheme=SequentialScheme(mnist.num_examples, batch_size=batch_size)))

cg = ComputationGraph(cost)
algorithm = GradientDescent(cost=cost, 
                            parameters=cg.parameters,
                            step_rule=Scale(learning_rate=learning_rate))


monitor = DataStreamMonitoring(variables=[cost],
                               data_stream=data_stream,
                               prefix="train")

main_loop = MainLoop(data_stream=data_stream, 
                     algorithm=algorithm,
                     extensions=[monitor, 
                                 FinishAfter(after_n_epochs=n_epochs), 
                                 Printing()]
            )

main_loop.run() 
