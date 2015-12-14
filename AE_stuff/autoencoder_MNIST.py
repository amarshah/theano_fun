# train a autoencoder for MNIST

import theano, blocks, fuel
import theano.tensor as T
import cPickle
from theano.tensor.shared_randomstreams import RandomStreams
from blocks.bricks import Rectifier, Logistic, MLP, Identity
from blocks.initialization import Uniform, Constant
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions import ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.graph import ComputationGraph
from blocks.extensions import FinishAfter, Printing
from blocks.bricks.cost import BinaryCrossEntropy, SquaredError
from fuel.datasets import BinarizedMNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten


srng = RandomStreams(seed=234)

n_vis = 28 ** 2
n_latent1 = 400
n_latent2 = 100

batch_size = 100
n_epochs = 4000
learning_rate = 0.02
save_freq = 20
model_save = 'AE_mnist_layer2.pkl'

x = T.matrix('features')

# Define the graph

mlp1 = MLP(activations=[Rectifier(), Rectifier()],
           dims=[n_latent1, n_latent2, n_latent1],
           biases_init=Constant(0.01),
           name='mlp1')

f = file('AE_mnist_layer1_model.pkl', 'rb')
model = cPickle.load(f) 
f.close()

import pdb; pdb.set_trace()
layer1_params = model.get_parameter_values()


#y = 
#y_hat = 
#reconstruct1 = BinaryCrossEntropy().apply(x, mlp1.apply(x))
reconstruct2 = SquaredError().apply(y, y_hat)

regularizer = 0
#for mlp in [mlp1, mlp2, mlp3, mlp4]:
for child in mlp1.children:
    for param in child.parameters:
        regularizer = regularizer + (param ** 2).sum()

cost = reconstruct1 + 0.1*regularizer
cost.name = 'reconstruct_error'
 
#------------------------------------------------------------------  
#for mlp in [mlp1, mlp2, mlp3, mlp4]:
mlp1._push_initialization_config()
for layer in mlp1.linear_transformations:
    layer.weights_init = Uniform(width=12. / (layer.input_dim + layer.output_dim))
mlp1.initialize()

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
