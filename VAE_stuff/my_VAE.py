__docformat__ = 'restructedtext en'

import cPickle, gzip, os, sys, timeit, numpy
import theano
import theano.tensor as T



class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b = None, activation = T.tanh ):
        self.input = input
        
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low  = -numpy.sqrt( 6. / (n_in + n_out) ),
                    high =  numpy.sqrt( 6. / (n_in + n_out) ),
                    size = (n_in, n_out)
                ),
                dtype = theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4
                
            W = theano.shared( value=W_values, name='W', borrow=True )

        if b is None:
            b_values = numpy.zeros( (n_out,), dtype = theano.config.floatX )
            b = theano.shared( value=b_values, name='b', borrow=True )

        self.W = W
        self.b = b

        lin_output = T.dot(input,self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        self.params = [self.W,self.b]


class GaussParamLayer(object):
    def __init__(self, rng, input, n_in, n_out=2, W=None, b=None ):
        self.input = input

        if W is None:
            W_values = numpy.asarray( 
                rng.uniform( 
                    low  = -numpy.sqrt( 6. / (n_in + 1) ),
                    high =  numpy.sqrt( 6. / (n_in + 1) ),
                    size = (n_in,n_out)
                ),
                dtype = theano.config.floatX
            )
                           
            W = theano.shared( value=W_values, name='W', borrow=True )

        if b is None:
            b_values = numpy.zeros( (n_out,), dtype = theano.config.floatX )
            b = theano.shared( value=b_values, name='b', borrow=True )
       
        self.W = W
        self.b = b

        lin_output = T.dot(input,self.W) + self.b
        self.mu    = lin_output[0] 
        self.lognu = lin_output[1]

        self.params = [W,b]


class Coder(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out ):
        self.HiddenLayer = HiddenLayer(
            rng, 
            input=input,
            n_in=n_in,
            n_out=n_hidden
        )
        
        self.GaussParamLayer = GaussParamLayer(
            rng,
            input=input,
            n_in=n_hidden
        )

        self.L2_square = (
            (self.HiddenLayer.W**2).sum() 
            + (self.HiddenLayer.b**2).sum()
            + (self.GaussParamLayer.W**2).sum()
            + (self.GaussParamLayer.b**2).sum()
        )

        self.params = self.HiddenLayer.params + self.GaussParamLayer.params

        self.input = input

        self.mu = self.GaussParamLayer.mu
        
        self.lognu = self.GaussParamLayer.lognu   
            

def load_data(dataset):
    f = gzip.open(dataset,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow = True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, 
                                               dtype=theano.config.floatX),
                                 borrow = borrow )
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow = borrow )
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval


def train_VAE(learning_rate = .01, L2_reg=1, n_epochs = 1000,
             dataset = 'mnist.pkl.gz', batch_size = 100, n_hidden=500, n_latent=20 ):

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size



    index = T.lscalar() 
    x = T.matrix('x')  
    h = T.ivector('h') 
                       

    rng = numpy.random.RandomState(1234)

    encoder = Coder(
        rng=rng,
        input=x,
        n_in=28*28,
        n_hidden=n_hidden,
        n_out= 2
    )

    decoder = Coder(
        rng=rng,
        input=encoder.mu,  #+ numpy.dot(numpy.sqrt(numpy.exp(encoder.lognu)),
                           #          normal(0,1,size=encoder.mu.shape)),
        n_in=n_latent,
        n_hidden=n_hidden,
        n_out=2
    )


    cost = (
        T.mean((0.5*(encoder.lognu-(encoder.mu**2)-T.exp(encoder.lognu)).sum()
        + T.exp(-0.5*T.log(2*3.14159) - 0.5*decoder.lognu - 0.5*(x-decoder.mu)**2/T.exp(decoder.lognu))))
        + L2_reg * (encoder.L2_square + decoder.L2_square)
    )    

    validate_model = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size]
        }
    )

    all_params = encoder.params + decoder.params

    gparams = [T.grad(cost,param) for param in all_params]

    updates = [
        (param, param - learning_rate*gparam )
        for param, gparam in zip(all_params, gparams)
    ]

    train_model = theano.function(
        inputs = [index],
        outputs = cost, 
        updates = updates,
        givens = {
            x : train_set_x[index*batch_size:(index + 1) * batch_size]
        }
    )       


    patience = 10000
    patience_increase = 2
    improvement_threshold = .995
    validation_frequency = min(n_train_batches, patience/2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0 
    done_looping = False

    while( epoch < n_epochs ) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange( n_train_batches ):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch-1)*n_train_batches + minibatch_index
            
            if (iter+1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    (
                        epoch,
                        minibatch_index+1,
                        n_train_batches,
                        this_validation_loss*100,
                    )
                )

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss*improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                           (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i.') %
          (best_validation_loss * 100., best_iter + 1 ))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    train_VAE()
    
