"""Learn a bijective autoencoder for generative modeling
Use random input partition, full invertible matrix and nonlinearity"""

import theano, numpy
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from blocks.algorithms import GradientDescent, Scale
from blocks.extensions import ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.extensions import FinishAfter, Printing
from fuel.datasets import BinarizedMNIST, MNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_batch, d_bucket, activation, activation_deriv,
                 w=None, index_permute=None, index_permute_reverse=None):
        srng = RandomStreams(seed=234)
        
        n_bucket = n_in / d_bucket + 1
        self.input = input

        # randomly permute input space
        if index_permute is None:
            index_permute = srng.permutation(n=n_in)#numpy.random.permutation(n_in)
            index_permute_reverse = T.argsort(index_permute)
            self.index_permute = index_permute
            self.index_permute_reverse = index_permute_reverse

        permuted_input = input[:, index_permute]
        self.permuted_input = permuted_input

        # initialize matrix parameters
        if w is None:
            bound = numpy.sqrt(3. / d_bucket)
            w_values = numpy.asarray(rng.uniform(low=-bound,
                                                 high=bound,
                                                 size=(n_bucket, d_bucket, d_bucket)),
                                     dtype=theano.config.floatX)
            w = theano.shared(value=w_values, name='w')
            
        self.w = w
        
        
        # compute outputs and Jacobians
        
        log_jacobian = T.alloc(0, n_batch)
        for b in xrange(n_bucket):
            bucket_size = d_bucket
            if b == n_bucket - 1:
                bucket_size = n_in - b * d_bucket
            
           
            if b>0:
                prev_input = x_b
                
                """here we warp the previous bucket of inputs and add to the new input"""            

            x_b = self.permuted_input[:, b*d_bucket:b*d_bucket + bucket_size]
            w_b = self.w[b, :bucket_size, :bucket_size]

            if b>0:
                x_b_plus = x_b + m_b
            else:
                x_b_plus = x_b

            Upper = T.triu(w_b)
            Lower = T.tril(w_b)
            Lower = T.extra_ops.fill_diagonal(Lower, 1.)
            log_det_Upper = T.log(T.abs_(T.nlinalg.ExtractDiag()(Upper))).sum() 

            W = T.dot(Upper, Lower)
            log_jacobian = log_jacobian + T.alloc(log_det_Upper, n_batch)

            lin_output_b = T.dot(x_b_plus, W)
            if b>0:
                lin_output = T.concatenate([lin_output, lin_output_b], axis=1)
            else:
                lin_output = lin_output_b
            if activation is not None:
                derivs = activation_deriv(lin_output_b)     
                #import pdb; pdb.set_trace()
                log_jacobian = log_jacobian + T.log(T.abs_(derivs)).sum(axis=1)                 
                    
        self.log_jacobian = log_jacobian        


        self.output = (
            lin_output[:, index_permute_reverse] if activation is None
            else activation(lin_output[:, index_permute_reverse])
        )

        self.params = [w]




class RectifiedLayer(object):
    def __init__(self, input, n_in, n_out, W=None, 

        
class FirstLayer(object):
    def __init__(self, input, rescale, recentre):
        srng = RandomStreams(seed=234)

        self.input = input

        dequantize_input = input + srng.uniform(size=input.shape, low=-0.5/255, high=0.5/255)

        self.output = rescale * (dequantize_input - recentre)


class FinalLayer(object):
    def __init__(self, input, n_in, n_batch, logv=None):
        self.input = input

        if logv is None:
            logv = theano.shared(value=numpy.zeros([1, n_in]), name='logv')
        
        self.logv = logv

        self.log_jacobian = T.alloc(logv.sum(), n_batch)  

#        import pdb; pdb.set_trace()
        self.output = T.tile(T.exp(logv), (n_batch, 1)) * input

        self.params = [logv]


class FullAE(object):
    def __init__(self, rng, input, rescale, recentre, n_in, n_batch, d_bucket, activation, activation_deriv):

        self.First = FirstLayer(input, rescale, recentre)

        self.Hidden1 = HiddenLayer(
            rng=rng,
            input=self.First.output,
            n_in=n_in,
            n_batch=n_batch,
            d_bucket=d_bucket,
            activation=activation,
            activation_deriv=activation_deriv
        )

        self.Hidden2 = HiddenLayer(
            rng=rng,
            input=self.Hidden1.output,
            n_in=n_in,
            n_batch=n_batch,
            d_bucket=d_bucket,
            activation=activation,
            activation_deriv=activation_deriv
        )

        self.Hidden3 = HiddenLayer(
            rng=rng,
            input=self.Hidden2.output,
            n_in=n_in,
            n_batch=n_batch,
            d_bucket=d_bucket,
            activation=activation,
            activation_deriv=activation_deriv
        )

        self.Hidden4 = HiddenLayer(
            rng=rng,
            input=self.Hidden3.output,
            n_in=n_in,
            n_batch=n_batch,
            d_bucket=d_bucket,
            activation=activation,
            activation_deriv=activation_deriv
        )

#        self.Final = FinalLayer(
#            input=self.Hidden2.output,
#            n_in=n_in,
#            n_batch=n_batch
#        )

        self.input = input
        
        self.output = self.Hidden4.output#Final.output

        self.log_jacobian = self.Hidden1.log_jacobian + self.Hidden2.log_jacobian \
            + self.Hidden3.log_jacobian + self.Hidden4.log_jacobian

        self.params = self.Hidden1.params + self.Hidden2.params \
            + self.Hidden3.params + self.Hidden4.params

        

def nonlinearity_deriv(input, a=0.75, b=1.5, c=0.5):
    ind = T.cast(T.le(input, c), theano.config.floatX)
    return ind * a + (1 - ind) * b

def nonlinearity(input, a=0.75, b=1.5, c=0.5):
    y1 = a * (input - c)
    y2 = b * (input - c)
    ind = T.cast(T.le(input, c), theano.config.floatX)
    return ind * y1 + (1 - ind) * y2


def train_AE(learning_rate=0.0001, n_epochs=1000, batch_size=100, d_bucket=250,
             activation=nonlinearity, activation_deriv=nonlinearity_deriv,
             model_save='AE2.pkl', save_freq=10):  


    rng = numpy.random.RandomState(1234)
    mnist = MNIST(("train",), sources=('features',))

    n_in = 28 ** 2

    
    x = T.matrix('features')

#    theano.config.compute_test_value = 'warn'    
#    x.tag.test_value = numpy.random.rand(100, 28**2).astype('float32') 
    
    AE = FullAE(
        rng=rng,
        input=x,
        rescale=1.,
        recentre=0.5,
        n_in=n_in,
        n_batch=batch_size,
        d_bucket=d_bucket,
        activation=activation,
        activation_deriv=activation_deriv)

    prior_term = (-0.5 * n_in * T.log(2 * 3.14159) - 0.5 * (AE.output ** 2).sum(axis=1)).mean() 
    prior_term.name = 'log_prior'
    log_jacobian = AE.log_jacobian.mean()
    log_jacobian.name = 'log_jacobian'
    cost = -(prior_term + log_jacobian)
    cost.name = 'negative_log_probability'

    params = AE.params
    output_max = (AE.output**2).sum(axis=1).max()
    output_max.name = 'output_max'
    output_min = (AE.output**2).sum(axis=1).min()
    output_min.name = 'output_min'

    h1_max = (AE.Hidden1.output**2).sum(axis=1).max()
    h1_max.name = 'h1_max'
    h1_min = (AE.Hidden1.output**2).sum(axis=1).min()
    h1_min.name = 'h1_min'

    h2_max = (AE.Hidden2.output**2).sum(axis=1).max()
    h2_max.name = 'h2_max'
    h2_min = (AE.Hidden2.output**2).sum(axis=1).min()
    h2_min.name = 'h2_min'



#    scales = AE.params[2]
#    scale_max = T.exp(scales.max())
#    scale_max.name = 'scale_max'
#    scale_min = T.exp(scales.min())
#    scale_min.name = 'scale_min'



    train_data_stream = Flatten(DataStream.default_stream(
        mnist,
        iteration_scheme=SequentialScheme(mnist.num_examples, batch_size=batch_size)))

#    epoch = train_data_stream.get_epoch_iterator()
#    for j, batch in enumerate(epoch):
#        import pdb; pdb.set_trace()
        


    monitor_data_stream = Flatten(DataStream.default_stream(
        mnist,
        iteration_scheme=SequentialScheme(mnist.num_examples, batch_size=batch_size)))

    algorithm = GradientDescent(cost=cost, 
                                parameters=params,
                                step_rule=Scale(learning_rate=learning_rate))


    monitor = DataStreamMonitoring(variables=[prior_term, log_jacobian, cost,
                                              h1_min, h1_max, h2_min, h2_max, output_min, output_max],
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
 
if __name__ == '__main__':
    train_AE()
            
            




