

import numpy, gzip, cPickle
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import sandbox, Out
from theano.scalar.basic_scipy import gammaln

theano.config.floatX = 'float32'

epsilon = 0.1
activation = T.tanh
n_hidden = 500
n_latent = 30
batch_size = 100
learning_rate = 0.0001
n_epochs = 400
n_vis = 28 ** 2

rng = numpy.random.RandomState(1234)
srng = RandomStreams(seed=234)

x = T.matrix('x')

# Define layers of encoder and decoder #####

# layer 1 of encoder

W_encoder_1_values = numpy.asarray(
    rng.uniform(
        low=-numpy.sqrt(6. / (n_vis + n_hidden)),
        high=numpy.sqrt(6. / (n_vis + n_hidden)),
        size=(n_vis, n_hidden)
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

W_encoder_2_values = numpy.asarray(
    rng.uniform(
        low=-numpy.sqrt(6. / (n_hidden + n_latent)),
        high=numpy.sqrt(6. / (n_hidden + n_latent)),
        size=(n_hidden, n_latent)
    ),
    dtype=theano.config.floatX
)
W_encoder_2 = theano.shared(value=W_encoder_2_values,
                             name='W_encoder_2',
                             borrow=True
                             )

b_encoder_2_values = numpy.zeros((n_latent,), dtype=theano.config.floatX)
b_encoder_2 = theano.shared(value=b_encoder_2_values,
                             name='b_encoder_2', borrow=True)

encoder_2 = T.dot(encoder_layer_1_out, W_encoder_2) + b_encoder_2

z = encoder_2 + epsilon * srng.uniform(size=encoder_2.shape,
                                       low=-1., high=1., dtype = encoder_2.dtype)
# T.exp(0.5 * encoder_lognu) * srng.normal(size=encoder_mu.shape,
#                                                          dtype=encoder_mu.dtype)


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
        low=-numpy.sqrt(6. / (n_hidden + n_vis)),
        high=numpy.sqrt(6. / (n_hidden + n_vis)),
        size=(n_hidden, n_vis)
    ),
    dtype=theano.config.floatX
)
W_decoder_p = theano.shared(value=W_decoder_p_values,
                            name='W_decoder_p',
                            borrow=True
                            )

b_decoder_p_values = numpy.zeros((n_vis, ), dtype=theano.config.floatX)
b_decoder_p = theano.shared(value=b_decoder_p_values,
                            name='b_decoder_p', borrow=True)

decoder_p = T.nnet.sigmoid(
    T.dot(decoder_layer_1_out, W_decoder_p) + b_decoder_p)


# Get the data #####

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_set_x, train_set_y = train_set
#import pdb; pdb.set_trace()
n_train_batches = train_set_x.shape[0] / batch_size
train_set_x = (train_set_x > 0.5).astype('float32')
              
#train_set_x = theano.shared(numpy.asarray(train_set_x > 0.5,
#                                          dtype=theano.config.floatX),
#                            borrow=True)
valid_set_x, valid_set_y = valid_set
n_valid_batches = valid_set_x.shape[0] / batch_size
valid_set_x = (valid_set_x > 0.5).astype('float32')

#valid_set_x = theano.shared(numpy.asarray(valid_set_x > 0.5,
#                                          dtype=theano.config.floatX),
#                            borrow=True)

#n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
#n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size




# Define cost and train and validate functions #####


log_ball_vol_D = n_latent * (0.5 * T.log(3.14159) 
                             + T.log(epsilon)) - T.log(gammaln(0.5 * n_latent + 1))

log_ball_vol_D_minus_1 = (n_latent - 1) * (0.5 * T.log(3.14159) 
                             + T.log(epsilon)) - T.log(gammaln(0.5 * n_latent + 0.5))

KL_term = T.cast(-log_ball_vol_D + 0.5 * T.log(2 * 3.14159) +
                  epsilon * n_latent / 3 * T.exp(log_ball_vol_D_minus_1 - log_ball_vol_D) *
                  (epsilon **2 + 3 * (encoder_2 * encoder_2).sum(axis=1)), 
                  theano.config.floatX)

#KL_divergence = T.cast(-0.5*(1 + encoder_lognu - encoder_mu**2
#                      - T.exp(encoder_lognu)).sum(axis=1), 'float32')
reconstruction_term = T.cast((x*T.log(decoder_p) +
                       (1-x)*T.log(1 - decoder_p)).sum(axis=1), 'float32')
cost = (KL_term - reconstruction_term).mean()

validate_model = theano.function(
    inputs=[x],
    outputs=cost
#    givens={
#        x: valid_set_x[index * batch_size:(index + 1) * batch_size]
#    },
)

encoder_params = [W_encoder_1, b_encoder_1,
                  W_encoder_2, b_encoder_2]
decoder_params = [W_decoder_1, b_decoder_1, W_decoder_p, b_decoder_p]
all_params = encoder_params + decoder_params

gparams = [T.grad(cost, param) for param in all_params]

updates = [
    (param, param - learning_rate*gparam)
    for param, gparam in zip(all_params, gparams)
]

train_model = theano.function(
    inputs=[x],
    outputs=cost,
    updates=updates
#    givens={
#        x: train_set_x[index*batch_size: (index + 1) * batch_size]
#    }
)

trainate_model = theano.function(
    inputs=[x],
    outputs=cost,
#    givens={
#        x: train_set_x[index*batch_size: (index + 1) * batch_size]
#    }
)

# Train the model and validate #####

patience = 10000
patience_increase = 2
improv_thresh = .995
valid_freq = min(n_train_batches, patience/2)

best_valid_negll = numpy.inf
best_iter = 0
test_score = 0.
#start_time = timeit.default_timer()

epoch = 0
done_looping = False
valid_neglls = numpy.zeros(n_valid_batches)

while(epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
  #  x = train_set_x[0:batch_size] 
  #  import pdb; pdb.set_trace()

    if epoch % 5 == 0:
        train_negll = trainate_model(train_set_x)
        this_train_negll = numpy.mean(train_negll)
        print(
            (
                epoch,
                this_train_negll
                )
        )

    for minibatch_index in xrange(n_train_batches):
        x = train_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
        minibatch_avg_cost = train_model(x)
        iter = (epoch-1)*n_train_batches + minibatch_index

        if (iter+1) % valid_freq == 0:
            for i in xrange(n_valid_batches):
                x = valid_set_x[i * batch_size: (i + 1) * batch_size]
                valid_neglls[i] = validate_model(x)

            this_valid_negll = numpy.mean(valid_neglls)

            # print(
            #     (
            #         epoch,
            #         minibatch_index+1,
            #         n_train_batches,
            #         this_valid_negll,
            #     )
            # )

            if this_valid_negll < best_valid_negll:
                if this_valid_negll < best_valid_negll*improv_thresh:
                    patience = max(patience, iter * patience_increase)

                best_valid_negll = this_valid_negll
                best_iter = iter

                # print(('    epoch %i, minibatch %i/%i, best model %f')
                #       % (epoch, minibatch_index + 1, n_train_batches, best_valid_negll))

                dict = {'W_encoder_1': W_encoder_1.get_value(),
                        'b_encoder_1': b_encoder_1.get_value(),
                        'W_encoder_2': W_encoder_2.get_value(),
                        'b_encoder_2': b_encoder_2.get_value(),
                        'W_decoder_1': W_decoder_1.get_value(),
                        'b_decoder_1': b_decoder_1.get_value(),
                        'W_decoder_p': W_decoder_p.get_value(),
                        'b_decoder_p': b_decoder_p.get_value()
                        }

                with open('/data/lisatmp3/shahamar/VAE_compress.pkl', 'w') as f:
                    cPickle.dump(dict, f)

                    

#        if patience <= iter:
#            done_looping = True
#            break

dict['final_train_nll'] = this_train_negll
with open('/data/lisatmp3/shahamar/VAE_compress.pkl', 'w') as f:
    cPickle.dump(dict, f)
