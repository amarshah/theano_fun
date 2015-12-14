
import numpy, theano, cPickle, gzip
import theano.tensor as T

from blocks.algorithms import GradientDescent, Adam, Momentum
from blocks.bricks import MLP, Rectifier, Logistic, Identity
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
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





class HadamardOp(theano.Op):
    __props__ = ()

#    import pdb; pdb.set_trace()
    
    def hadamard(self, v):
        n_in = v.shape[1]
        n = n_in
        while n > 1:
            for n_start in xrange(0, n_in, n):
                first_half = numpy.sqrt(0.5) * v[:,n_start : (n_start + n/2)]
                second_half = numpy.sqrt(0.5) * v[:,(n_start + n/2) : (n_start + n)]
                v[:,n_start : (n_start + n/2)] = first_half + second_half
                v[:,(n_start + n/2) : (n_start + n)] = first_half - second_half
            n = n/2
        return v
#        if n_in>1:
#            v1 = self.hadamard(v[:, :n_in/2])
#            v2 = self.hadamard(v[:, n_in/2:])                
#            return numpy.sqrt(0.5) * numpy.concatenate((v1 + v2, v1 - v2), axis=1)
#        else:
#            return v

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = self.hadamard(x)

    def grad(self, inputs, output_grads):
        return [HadamardOp()(output_grads)]


# n_hidden is number of complex numbers
class HiddenAndInputToHidden(object):
    def __init__(self, srng, data_input, hidden_input, n_data, n_hidden, n_diagonals,
                 index_permute, theta, V_re, V_im, log_bias, Hadamard, Permutation):

        self.index_permute = index_permute

        if not ((n_hidden - 1) & n_hidden) == 0:
            print 'hidden input dimension must be power of 2!'       
        

        # operates on the concatenation of real and imaginary parts
        def vec_times_diag(vec, n_hidden, diag):
            #import pdb; pdb.set_trace()
            x = vec[:, :n_hidden]
            y = vec[:, n_hidden:]

            Re = T.nlinalg.AllocDiag()(T.cos(diag))
            Im = T.nlinalg.AllocDiag()(T.sin(diag))

            x_Re = T.dot(x, Re)
            x_Im = T.dot(x, Im)
            y_Re = T.dot(y, Re)
            y_Im = T.dot(y, Im)

            output = T.concatenate([x_Re - y_Im, x_Im + y_Re], axis=1)

            return output
            
        # operates on the concatenation of real and imaginary parts
        def vec_permutation(vec, n_hidden, index_permute):
            x = vec[:, :n_hidden]
            y = vec[:, n_hidden:]

            x_permute = x[:, index_permute]
            y_permute = y[:, index_permute]

            output = T.concatenate([x_permute, y_permute], axis=1)
            return output

        
        self.Hadamard = Hadamard

        import pdb;

        # first take linear transform of hidden RNN state        
        step1 = vec_times_diag(hidden_input, n_hidden, theta[0,:])
        step2 = T.concatenate([T.dot(step1[:, :n_hidden], Hadamard),
                               T.dot(step1[:, n_hidden:], Hadamard)], axis=1) 
#        step3 = vec_permutation(step2, n_hidden, index_permute)
        step3 = T.concatenate([T.dot(step2[:, :n_hidden], Permutation),
                               T.dot(step2[:, n_hidden:], Permutation)], axis=1)
        step4 = vec_times_diag(step3, n_hidden, theta[1,:])
        step5 = T.concatenate([T.dot(step4[:, :n_hidden], Hadamard),
                               T.dot(step4[:, n_hidden:], Hadamard)], axis=1)
        step6 = vec_times_diag(step5, n_hidden, theta[2,:])
        hidden_lin_output = step6

        # second take linear transform of data at time T
        data_lin_output_re = T.dot(data_input, V_re)
        data_lin_output_im = T.dot(data_input, V_im)
        pdb.set_trace()
        data_lin_output = T.concatenate([data_lin_output_re, data_lin_output_im], axis=0)


        # sum lin outputs of linearly transformed hidden and data states 
        lin_output = hidden_lin_output + data_lin_output
                                        
        self.params = [theta, V_re, V_im]

        lin_output_re = lin_output[:, :n_hidden]
        lin_output_im = lin_output[:, n_hidden:]

        self.params = self.params + [log_bias] 

        modulus = T.sqrt(lin_output_re ** 2 + lin_output_im ** 2)
        scale = T.maximum(modulus - T.exp(log_bias), 0.) / (modulus + 1e-5)

        nonlin_output_re = lin_output_re * scale
        nonlin_output_im = lin_output_im * scale

        nonlin_output = T.concatenate([nonlin_output_re, 
                                       nonlin_output_im], axis=1)

        self.output = nonlin_output

        
class HiddenToOutput(object):
    def __init__(self, hidden_input, n_hidden, n_output, U, out_bias, activation=None):
        self.params = [U, out_bias]    
        
        #import pdb; pdb.set_trace()
        lin_output = T.dot(hidden_input, U) + out_bias.dimshuffle('x',0)

        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        


# assumes single output at the end    
class ComplexRNN(object):
    def __init__(self, srng, inputs, outputs, time_steps, n_data, n_hidden, n_output, n_batch,
                 output_activation, n_diagonals=3, hidden_state_start=None, index_permute=None,
                 theta=None, V_re=None, V_im=None, log_bias=None, U=None, out_bias=None,
                 Hadamard=None, Permutation=None):

        if hidden_state_start is None:
            # hidden state initialized to length 2*n_hidden
            # first n_hidden entries are real parts, next n_hidden entries are imaginary parts
            hidden_state_start_values = numpy.zeros(shape=(1,2*n_hidden), dtype=theano.config.floatX)
            hidden_state_start = theano.shared(value=hidden_state_start_values, name='hidden_state_start')

        hidden_state_start_batch = T.tile(hidden_state_start, [n_batch, 1])
        
        if index_permute is None:
            index_permute = srng.permutation(n=n_hidden)

        if theta is None:
            theta_values = numpy.asarray(rng.uniform(low=-numpy.pi,
                                                     high=numpy.pi,
                                                     size=(n_diagonals, n_hidden)),
                                         dtype=theano.config.floatX)
            theta = theano.shared(value=theta_values, name='theta')           

        if V_re is None:
            bin = numpy.sqrt(6. / (n_data + n_hidden))
            V_re_values = numpy.asarray(rng.uniform(low=-bin,
                                                    high=bin,
                                                    size=(n_data, n_hidden)),
                                        dtype=theano.config.floatX)
            V_re = theano.shared(value=V_re_values, name='V_re')

        if V_im is None:
            bin = numpy.sqrt(6. / (n_data + n_hidden))
            V_im_values = numpy.asarray(rng.uniform(low=-bin,
                                                    high=bin,
                                                    size=(n_data, n_hidden)),
                                        dtype=theano.config.floatX)
            V_im = theano.shared(value=V_im_values, name='V_im')

        if log_bias is None:
            log_bias_values = numpy.ones(shape=(n_hidden,), dtype=theano.config.floatX)
            log_bias = theano.shared(value= -100*log_bias_values, name='out_bias')

            log_bias = theano.shared(value=numpy.asarray(-100, dtype=theano.config.floatX),
                                     name='log_bias')            

        if U is None:
            bin = numpy.sqrt(6. / (2*n_hidden + n_output))
            U_values = numpy.asarray(rng.uniform(low=-bin,
                                                 high=bin,
                                                 size=(2 * n_hidden, n_output)),
                                     dtype=theano.config.floatX)
            U = theano.shared(value=U_values, name='U')

        if out_bias is None:
            out_bias_values = numpy.zeros(shape=(n_output,), dtype=theano.config.floatX)
            out_bias = theano.shared(value=out_bias_values, name='out_bias')

        if Hadamard is None:
            def sethadamard(n):
                if n==1:
                    return numpy.array([[1]], dtype=theano.config.floatX)
                else:
                    H = sethadamard(n/2)
                    col1 = numpy.concatenate((H, H), axis=0)
                    col2 = numpy.concatenate((H, -H), axis=0)
                    return numpy.sqrt(1./2) * numpy.concatenate((col1, col2), axis=1)
            Hadamard_values = sethadamard(n_hidden)
            Hadamard = theano.shared(value=Hadamard_values, name='Hadamard')

        if Permutation is None:
            Permutation = T.zeros_like(Hadamard)
            perm = numpy.random.permutation(n_hidden)
            for i in xrange(n_hidden):
                Permutation = T.set_subtensor(Permutation[perm[i], i], 1)                     

        def recurrence(x, h):
            Step = HiddenAndInputToHidden(srng=srng,
                                          data_input=x,
                                          hidden_input=h,
                                          n_data=n_data,
                                          n_hidden=n_hidden,
                                          n_diagonals=3,
                                          index_permute=index_permute,
                                          theta=theta,
                                          V_re=V_re,
                                          V_im=V_im,
                                          log_bias=log_bias,
                                          Hadamard=Hadamard,
                                          Permutation=Permutation)

            return Step.output

        
        import pdb; pdb.set_trace()
        hidden_states, updates = theano.scan(fn=recurrence,
                                             sequences=inputs,
                                             outputs_info=hidden_state_start_batch)
        
#        f = theano.function(inputs=[inputs], outputs=hidden_states[-1,:,:], updates=updates)         
        
        final_hidden_state = hidden_states[-1,:,:]
        

        Final = HiddenToOutput(final_hidden_state, 
                           n_hidden,
                           n_output,
                           U,
                           out_bias,
                           activation=output_activation)


        self.output = Final.output
        self.params = [U, V_re, V_im, log_bias, hidden_state_start, theta] # out_bias


##############
n_epochs = 100
model_save = 'complex_rnn_mnist.pkl'
save_freq = 1
batch_size = 1
rng = numpy.random.RandomState(1234) 
srng = T.shared_randomstreams.RandomStreams(seed=234)
learning_rate = 0.001

(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = cPickle.load(gzip.open("mnist.pkl.gz", 'rb'))

train_x = numpy.reshape(train_x, (train_x.shape[1], train_x.shape[0], 1))

x = T.matrix('x')
y = T.matrix('y')


x.tag.test_value = numpy.random.rand(28**2,10,1).astype('float32') 
y.tag.test_value = numpy.random.randint(0, 10, size=(10,1)).astype('float32')
 

RNN = ComplexRNN(srng=srng,
                 inputs=x, 
                 outputs=y,
                 time_steps=2,
                 n_data=1,
                 n_hidden=128,  
                 n_output=10,
                 n_batch=batch_size,
                 output_activation=T.nnet.softmax,
                 n_diagonals=3)

parameters = RNN.params

one_hot = T.extra_ops.to_one_hot(T.cast(y, 'int32'),
                                 10,
                                 dtype='int32')

cost = T.nnet.categorical_crossentropy(RNN.output, one_hot).mean()
cost.name = 'cross_entropy'


gradients = [T.grad(cost, p) for p in parameters]

quick_cost = theano.function([x, y], cost)
compute_gradients = theano.function([x, y], gradients) 

updates = dict((p, p - learning_rate * g) for p, g in zip(parameters, gradients))
cost_and_update = theano.function([x, y], cost, updates=updates)

import pdb; pdb.set_trace()

for i in range(len(train_x)):
    print cost_and_update([train_x[:,i,:], train_y[i]])

















#####################


mnist_train = MNIST(('train',))#, sources=('features', 'targets'))
num_examples = 10#mnist_train.num_examples

train_data_stream = Flatten(DataStream.default_stream(
    mnist_train,
    iteration_scheme=ShuffledScheme(num_examples, batch_size=batch_size)))

train_monitor_stream = Flatten(DataStream.default_stream(
    mnist_train,
    iteration_scheme=ShuffledScheme(num_examples, batch_size=batch_size)))


x = T.matrix('features')
y = T.matrix('targets')

epoch = train_data_stream.get_epoch_iterator()
for j, batch in enumerate(epoch):
    if j>0:
        break
    theano.config.compute_test_value = 'warn'    
    x.tag.test_value = batch[0]#numpy.random.rand(100, 28**2).astype('float32') 
    y.tag.test_value = batch[1]#numpy.random.randint(0, 10, size=(100,1)).astype('float32')
 






gradients = T.grad(cost, RNN.params)
gradients.name = 'gradients'
import pdb; pdb.set_trace()

#theano.pp(cost )


#epoch = train_data_stream.get_epoch_iterator()
#for j, batch in enumerate(epoch):
#import pdb; pdb.set_trace()


#valid_monitor_stream = Flatten(DataStream.default_stream(
#    mnist_valid,
#    iteration_scheme=ShuffledScheme(mnist_valid.num_examples, batch_size=batch_size)))

step_rule = Momentum(learning_rate=1e-4, momentum=0.5)
#step_rule = Adam(learning_rate=0.001, beta1=0.9, beta2=0.01, epsilon=0.0001)
algorithm = GradientDescent(cost=cost, 
                            parameters=RNN.params,
                            step_rule=step_rule)

monitor_variables = [cost, gradients]
model = Model(cost)

#checkpoint = Checkpoint(model_save, after_training=False)
#checkpoint.add_condition(['after_epoch'],
#                         predicate=OnLogRecord('valid_negative_ll_best_so_far'))

#list = [algorithm.steps[param].norm(2).copy(name="step_norm:%s" % name)
#        for name, param in model.get_parameter_dict().items()]

#train_data_monitoring = TrainingDataMonitoring(list,
#                                               prefix="train",
#                                               after_epoch=True) # prints L2 of gradient steps

main_loop = MainLoop(model=model,
                     data_stream=train_data_stream, 
                     algorithm=algorithm,
                     extensions=[Timing(),
                                 DataStreamMonitoring(monitor_variables,
                                                      train_monitor_stream,
                                                      prefix="train"),
#                                 DataStreamMonitoring(
#                                     monitor_variables, valid_monitor_stream, prefix="valid"),
                                 FinishAfter(after_n_epochs=n_epochs),
#                                 TrackTheBest('valid_negative_ll'),
                                 ProgressBar(),
                                 Checkpoint(model_save, every_n_epochs=save_freq),
#                                 train_data_monitoring,
                                 Printing()
                     ]
            )

main_loop.run() 

        


    
