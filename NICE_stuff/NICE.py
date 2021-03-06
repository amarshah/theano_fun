
import numpy, theano
import theano.tensor as T

from blocks.algorithms import GradientDescent, Adam, Momentum
from blocks.bricks import MLP, Rectifier, Logistic, Identity
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.predicates import OnLogRecord
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.training import TrackTheBest
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.select import Selector
from fuel.datasets import BinarizedMNIST, MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Flatten



batch_size = 100
model_save = 'nice_mnist.pkl'
save_freq = 10
n_epochs = 1000


class NICE(object):
    def __init__(self, rng, srng, input, index1, index2, n_in1, n_in2, n_hidden_layers, d_hidden, n_batch):
        self.input = input
        
        dequantize_input = input + srng.uniform(size=input.shape, low=-0.5/255, high=0.5/255)

        input1 = input[:, index1]
        input2 = input[:, index2]

        Hidden1 = HiddenLayer(rng, input1, input2, n_in1, n_in2, n_hidden_layers, d_hidden)
        Hidden2 = HiddenLayer(rng, Hidden1.output2, Hidden1.output1, n_in2, n_in1, n_hidden_layers, d_hidden)
        Hidden3 = HiddenLayer(rng, Hidden2.output2, Hidden2.output1, n_in1, n_in2, n_hidden_layers, d_hidden)
#        Hidden4 = HiddenLayer(rng, Hidden3.output2, Hidden3.output1, n_in2, n_in1, n_hidden_layers, d_hidden)        
        H = T.concatenate([Hidden3.output1, Hidden3.output2], axis=1)
        Final = ScaleLayer(rng, H, n_in1 + n_in2, n_batch)

        self.output = Final.output

        self.log_jacobian = Final.log_jacobian

        self.params = Hidden1.params + Hidden2.params + \
            Hidden3.params + Final.params




class ScaleLayer(object):
    def __init__(self, rng, input, n_in, n_batch, log_scale=None):
        self.input = input
        
        if log_scale is None:
            log_scale_values = numpy.asarray(rng.uniform(low=-0.1,
                                                         high=0.1,
                                                         size=(1, n_in)),
                                             dtype=theano.config.floatX)
            log_scale = theano.shared(value=log_scale_values, name='log_scale')

        self.output = input * T.tile(T.exp(log_scale), (n_batch, 1)) 
            
        self.params = [log_scale]
        #import pdb; pdb.set_trace()
        self.log_jacobian = log_scale.sum()


class HiddenLayer(object):
    def __init__(self, rng, input1, input2, n_in1, n_in2, n_hidden_layers, d_hidden):
        self.input1 = input1
        self.input2 = input2
        
        CouplingFunc = WarpNetwork(rng, input1, n_hidden_layers, d_hidden, n_in1, n_in2)  
        
        self.output1 = input1
        self.output2 = input2 + CouplingFunc.output

        self.params = CouplingFunc.params


class WarpNetwork(object):
    def __init__(self, rng, input, n_hidden_layers, d_hidden, n_in, n_out):        
        self.input = input
        output = input
        
        params = []

        n_in_layer = n_in
        for layer in xrange(n_hidden_layers):
            Warp = Layer(rng, output, n_in_layer, d_hidden, activation=T.nnet.relu)
            output = Warp.output
            n_in_layer = d_hidden
            params = params + Warp.params

        Warp = Layer(rng, output, d_hidden, n_out)
        output = Warp.output
        self.output = output

        self.params = params + Warp.params
        
        


class Layer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=None):
        self.input = input

        if W is None:            
            bin = numpy.sqrt(6. / (n_in + n_out))
            
            W_values = numpy.asarray(rng.uniform(low=-bin,
                                                 high=bin,
                                                 size=(n_in, n_out)),
                                     dtype = theano.config.floatX)
            W = theano.shared(value=W_values, name='W')
        
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        lin_output = T.dot(input, W) + b
               
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        self.params = [W, b]


rng = numpy.random.RandomState(1234) 
srng = T.shared_randomstreams.RandomStreams(seed=234)

mnist_train = BinarizedMNIST(('train',), sources=('features',))
mnist_valid = BinarizedMNIST(('valid',), sources=('features',))
x = T.matrix('features')

#import pdb; pdb.set_trace()

#########
theano.config.compute_test_value = 'warn'    
x.tag.test_value = numpy.random.rand(100, 28**2).astype('float32') 
#########

index1 = range(0, 28**2, 2)
index2 = range(1, 28**2, 2)

n_in1 = len(index1)
n_in2 = len(index2)

NICE = NICE(rng, srng, x, index1, index2, n_in1, n_in2, n_hidden_layers=2, d_hidden=500, n_batch=batch_size) 

prior_term = - ((T.nnet.softplus(NICE.output) + T.nnet.softplus(- NICE.output)).sum(axis=1)).mean()
prior_term.name = 'prior_term'
jacobian_term = NICE.log_jacobian
jacobian_term.name = 'jacobian_term'
cost = - (prior_term + jacobian_term) 
cost.name = 'negative_ll'


train_data_stream = Flatten(DataStream.default_stream(
    mnist_train,
    iteration_scheme=ShuffledScheme(mnist_train.num_examples, batch_size=batch_size)))

train_monitor_stream = Flatten(DataStream.default_stream(
    mnist_train,
    iteration_scheme=ShuffledScheme(mnist_train.num_examples, batch_size=batch_size)))

valid_monitor_stream = Flatten(DataStream.default_stream(
    mnist_valid,
    iteration_scheme=ShuffledScheme(mnist_valid.num_examples, batch_size=batch_size)))

#step_rule = Adam(learning_rate=0.001, beta1=0.9, beta2=0.01, epsilon=0.0001)
step_rule = Momentum(learning_rate=0.001, momentum=0.5)
algorithm = GradientDescent(cost=cost, 
                            parameters=NICE.params,
                            step_rule=step_rule)


monitor_variables = [cost]
#monitor = DataStreamMonitoring(variables=[cost, prior_term, jacobian_term],
#                               data_stream=valid_monitor_stream,
#                               prefix="train")

model = Model(cost)

checkpoint = Checkpoint(path=model_save, after_training=False)
checkpoint.add_condition(['after_epoch'],
                         predicate=OnLogRecord('valid_negative_ll_best_so_far'))

main_loop = MainLoop(model=model,
                     data_stream=train_data_stream, 
                     algorithm=algorithm,
                     extensions=[Timing(),
                                 DataStreamMonitoring(
                                     monitor_variables, train_monitor_stream, prefix="train"),
                                 DataStreamMonitoring(
                                     monitor_variables, valid_monitor_stream, prefix="valid"),
                                 FinishAfter(after_n_epochs=n_epochs),
                                 TrackTheBest('valid_negative_ll'),
                                 Printing(),
                                 ProgressBar(),
                                 checkpoint]
            )

main_loop.run() 
