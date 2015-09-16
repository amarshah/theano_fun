

import numpy, os, sys, timeit, gzip, cPickle
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

activation = T.tanh
n_hidden = 500
n_latent = 20
batch_size = 100
learning_rate = 0.05
n_epochs = 1000

rng = numpy.random.RandomState(1234)
srng = RandomStreams(seed=234)
x = T.matrix('x')
z = T.vector('z')

# Define layers of encoder and decoder #####

# layer 1 of encoder

W_encoder_1_values = numpy.asarray(
    rng.uniform(
        low=-numpy.sqrt(6. / (28*28 + n_hidden)),
        high=numpy.sqrt(6. / (28*28 + n_hidden)),
        size=(28*28, n_hidden)
    ),
    dtype=theano.config.floatX
)
W_encoder_1 = theano.shared(value=W_encoder_1_values,
                            name='W_encoder_1',
                            borrow=True
                            )

b_encoder_1_values = numpy.zeros((n_hidden,), dtype=theano.config.floatX)
b_encoder_1 = theano.shared(value=b_encoder_1_values,
                            name='b_encoder_1', borrow=True)

encoder_layer_1_out = activation(T.dot(x, W_encoder_1) + b_encoder_1)

# layer 2 of encoder

W_encoder_mu_values = numpy.asarray(
    rng.uniform(
        low=-numpy.sqrt(6. / (n_hidden + 1)),
        high=numpy.sqrt(6. / (n_hidden + 1)),
        size=(n_hidden, 1)
    ),
    dtype=theano.config.floatX
)
W_encoder_mu = theano.shared(value=W_encoder_mu_values,
                             name='W_encoder_mu',
                             borrow=True
                             )

b_encoder_mu_values = numpy.zeros((1,), dtype=theano.config.floatX)
b_encoder_mu = theano.shared(value=b_encoder_mu_values,
                             name='b_encoder_mu', borrow=True)

encoder_mu = T.dot(encoder_layer_1_out, W_encoder_mu) + b_encoder_mu

W_encoder_lognu_values = numpy.asarray(
    rng.uniform(
        low=-numpy.sqrt(6. / (n_hidden + 1)),
        high=numpy.sqrt(6. / (n_hidden + 1)),
        size=(n_hidden, 1)
    ),
    dtype=theano.config.floatX
)

W_encoder_lognu = theano.shared(value=W_encoder_lognu_values,
                                name='W_encoder_lognu',
                                borrow=True
                                )

b_encoder_lognu_values = numpy.zeros((1,), dtype=theano.config.floatX)
b_encoder_lognu = theano.shared(value=b_encoder_lognu_values,
                                name='b_encoder_lognu', borrow=True)

encoder_lognu = T.dot(encoder_layer_1_out, W_encoder_lognu) + b_encoder_lognu


# layer 1 of decoder

W_decoder_1_values = numpy.asarray(
    rng.uniform(
        low=-numpy.sqrt(6. / (n_latent + n_hidden)),
        high=numpy.sqrt(6. / (n_latent + n_hidden)),
        size=(n_latent, n_hidden)
    ),
    dtype=theano.config.floatX
)
W_decoder_1 = theano.shared(value=W_decoder_1_values,
                            name='W_decoder_1',
                            borrow=True
                            )

b_decoder_1_values = numpy.zeros((n_hidden,), dtype=theano.config.floatX)
b_decoder_1 = theano.shared(value=b_decoder_1_values,
                            name='b_decoder_1', borrow=True)

decoder_layer_1_out = activation(T.dot(z, W_decoder_1) + b_decoder_1)

# layer 2 of decoder

W_decoder_p_values = numpy.asarray(
    rng.uniform(
        low=-numpy.sqrt(6. / (n_hidden + 1)),
        high=numpy.sqrt(6. / (n_hidden + 1)),
        size=(n_hidden, 1)
    ),
    dtype=theano.config.floatX
)
W_decoder_p = theano.shared(value=W_decoder_p_values,
                            name='W_decoder_p',
                            borrow=True
                            )

b_decoder_p_values = numpy.zeros((1, ), dtype=theano.config.floatX)
b_decoder_p = theano.shared(value=b_decoder_p_values,
                            name='b_decoder_p', borrow=True)

decoder_p = T.nnet.sigmoid(
    T.dot(decoder_layer_1_out, W_decoder_p) + b_decoder_p)


# Get the data #####

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_set_x, train_set_y = train_set
train_set_x = theano.shared(numpy.asarray(train_set_x > 0.5,
                                          dtype=theano.config.floatX),
                            borrow=True)
valid_set_x, valid_set_y = valid_set
valid_set_x = theano.shared(numpy.asarray(valid_set_x > 0.5,
                                          dtype=theano.config.floatX),
                            borrow=True)

n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size


# Define cost and train and validate functions #####

index = T.lscalar()

KL_divergence = -0.5*(1 + encoder_lognu - encoder_mu**2
                      - T.exp(encoder_lognu)).sum(axis=1)
z = encoder_mu + T.sqrt(T.exp(encoder_lognu))*srng.normal(encoder_lognu.shape)
reconstruction_term = (x*T.log(decoder_p) +
                       (1-x)*T.log(decoder_p)).sum(axis=1)
cost = (KL_divergence - reconstruction_term).mean()

validate_model = theano.function(
    inputs=[index],
    outputs=cost,
    givens={
        x: valid_set_x[index * batch_size:(index + 1) * batch_size]
    },
)

encoder_params = [W_encoder_1, b_encoder_1,
                  W_encoder_mu, b_encoder_mu, W_encoder_lognu, b_encoder_lognu]
decoder_params = [W_decoder_1, b_decoder_1, W_decoder_p, b_decoder_p]
all_params = encoder_params + decoder_params

gparams = [T.grad(cost, param) for param in all_params]

updates = [
    (param, param - learning_rate*gparam)
    for param, gparam in zip(all_params, gparams)
]

train_model = theano.function(
    inputs=[index],
    outputs=cost,
    updates=updates,
    givens={
        x: train_set_x[index*batch_size: (index + 1) * batch_size]
    }
)

# Train the model and validate #####

patience = 10000
patience_increase = 2
improv_thresh = .995
valid_freq = min(n_train_batches, patience/2)

best_valid_negll = numpy.inf
best_iter = 0
test_score = 0.
start_time = timeit.default_timer()

epoch = 0
done_looping = False

while(epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in xrange(n_train_batches):
        minibatch_avg_cost = train_model(minibatch_index)
        iter = (epoch-1)*n_train_batches + minibatch_index

        if (iter+1) % valid_freq == 0:
            valid_neglls = [validation_model(i)
                            for i in xrange(n_valid_batches)]
            this_valid_negll = numpy.mean(valid_neglls)

            print(
                (
                    epoch,
                    minibatch_index+1,
                    n_train_batchs,
                    -this_valid_negll,
                )
            )

            if this_valid_negll < best_valid_negll:
                if this_valid_negll < best_valid_negll*improv_thresh:
                    patience = max(patience, iter * patience_increase)

                best_valid_negll = this_valid_negll
                best_iter = iter

                print(('    epoch %i, minibatch %i/%i, best model %f %%')
                      % (epoch, minibatch_index + 1, n_train_batches))

        if patience <= iter:
            done_looping = True
            break

end_time = timeit.default.timer()
print(('Optimization complete. Best validation loglik of %f %% '
       'obtained at iteration i.') %
      (-best_valid_negll, best_iter + 1))
print >> sys.stderr, ('The code for file ' +
                      os.path.split(__file__)[1] +
                      ' ran for %.2fm' % ((end_time - start_time) / 60.))
