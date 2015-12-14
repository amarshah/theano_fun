import theano
import pdb
from fftconv import cufft, cuifft
import numpy as np
import theano.tensor as T
from theano.ifelse import ifelse

def times_stuff(input, v, u):
    return T.outer(T.dot(input, v), v) 

np.random.seed(1234)
rng = np.random.RandomState(1234)

# x =  T.matrix()
# n_b = 5
# n_h = 10

# theano.config.compute_test_value = 'warn'
# x.tag.test_value = np.asarray(np.random.rand(n_b, n_h)).astype('float32') 

# bin = np.sqrt(6. / (n_h + 1))
# v_values = np.asarray(rng.uniform(low=-bin,
#                                   high=bin,
#                                   size=(n_h,)),
#                       dtype=theano.config.floatX)
# v = theano.shared(v_values)
# u_values = np.asarray(rng.uniform(low=-bin,
#                                   high=bin,
#                                   size=(n_h,)),
#                       dtype=theano.config.floatX)
# u = theano.shared(u_values)

# U = theano.shared(np.asarray(rng.uniform(low=-1., high=1., size=(n_h,)),
#                              dtype=theano.config.floatX))

# y = times_stuff(x, v, u)

# cost = T.dot(y, U)
# cost = cost.mean()

# dcdy = T.tile(U.dimshuffle('x',0), [n_b, 1])

# ########compute deriv

# dv = (dcdy * T.dot(x, v).dimshuffle(0,'x') + x * T.dot(dcdy, v).dimshuffle(0, 'x')).mean(axis=0)
# #du = (dcdy * T.dot(x, u).dimshuffle(0,'x') + T.dot(dcdy, u).dimshuffle(0,'x') * x).mean(axis=0)

# #du = -2*( T.outer(T.dot(dcdy, u)*T.dot(x, v), v) / ((u**2).sum()**2) ).mean(axis=0)
# #uTu = T.dot(u.T, u)
# #du = -2./ (uTu**2) * (T.outer((dcdy * T.outer(T.dot(x, v), v)).sum(axis=1), u)).mean(axis=0)

# actual_dv = T.grad(cost, v)

# import pdb; pdb.set_trace()



def times_reflection(input, n_hidden, reflection):
    input_re = input[:, :n_hidden]
    input_im = input[:, n_hidden:]
    
    reflect_re = reflection[:n_hidden]
    reflect_im = reflection[n_hidden:]

    vstarv = (reflect_re**2 + reflect_im**2).sum()

    input_re_reflect = input_re - 2. / vstarv * (T.outer(T.dot(input_re, reflect_re), reflect_re) 
                                                 + T.outer(T.dot(input_re, reflect_im), reflect_im) 
                                                 - T.outer(T.dot(input_im, reflect_im), reflect_re) 
                                                 + T.outer(T.dot(input_im, reflect_re), reflect_im)) 
    
    input_im_reflect = input_im - 2. / vstarv * (T.outer(T.dot(input_im, reflect_re), reflect_re) 
                                                 + T.outer(T.dot(input_im, reflect_im), reflect_im) 
                                                 + T.outer(T.dot(input_re, reflect_im), reflect_re) 
                                                 - T.outer(T.dot(input_re, reflect_re), reflect_im)) 

    return T.concatenate([input_re_reflect, input_im_reflect], axis=1)      


np.random.seed(1234)
rng = np.random.RandomState(1234)


x = T.matrix()
n_hidden = 10
n_batch = 5

theano.config.compute_test_value = 'warn'
x.tag.test_value = np.random.rand(n_batch, 2*n_hidden).astype('float32') 



bin = np.sqrt(6. / (2*n_hidden + 1))
v_values = np.asarray(rng.uniform(low=-bin,
                                  high=bin,
                                  size=(1, 2*n_hidden)),
                      dtype=theano.config.floatX)
v = theano.shared(v_values)

U = theano.shared(np.asarray(rng.uniform(low=-1., high=1., size=(2*n_hidden,)),
                             dtype=theano.config.floatX))

y = times_reflection(x, n_hidden, v[0])

costs = T.dot(y, U)
c = T.mean(costs)


dy = T.tile(U.dimshuffle('x',0), [n_batch, 1])

###----- Gradient computation ---------------------------------------------------------------------

v_re = v[:, :n_hidden]
v_im = v[:, n_hidden:]
vstarv = (v_re ** 2 + v_im ** 2).sum()

dstep7_re = dy[:, :n_hidden]
dstep7_im = dy[:, n_hidden:]
step6_re = x[:, :n_hidden]
step6_im = x[:, n_hidden:]
v_re_dot_v_re = T.dot(v_re, v_re.T)
v_im_dot_v_im = T.dot(v_im, v_im.T)
v_im_dot_v_re = T.dot(v_im, v_re.T)

dstep7_re_dot_v_re = T.addbroadcast(T.dot(dstep7_re, v_re.T), 1) #n_b x 1
dstep7_re_dot_v_im = T.addbroadcast(T.dot(dstep7_re, v_im.T), 1)
step6_re_dot_v_re = T.addbroadcast(T.dot(step6_re, v_re.T), 1)
step6_re_dot_v_im = T.addbroadcast(T.dot(step6_re, v_im.T), 1)
dstep7_im_dot_v_re = T.addbroadcast(T.dot(dstep7_im, v_re.T), 1)
dstep7_im_dot_v_im = T.addbroadcast(T.dot(dstep7_im, v_im.T), 1)
step6_im_dot_v_re = T.addbroadcast(T.dot(step6_im, v_re.T), 1)
step6_im_dot_v_im = T.addbroadcast(T.dot(step6_im, v_im.T), 1)

dstep7_re_timesum_step6_re = (dstep7_re * step6_re).sum(axis=1)
dstep7_re_timesum_step6_im = (dstep7_re * step6_im).sum(axis=1)
dstep7_im_timesum_step6_re = (dstep7_im * step6_re).sum(axis=1)
dstep7_im_timesum_step6_im = (dstep7_im * step6_im).sum(axis=1)

#--------

dstep7_re_RedOpdv_re_term1 = - 2. / vstarv * (dstep7_re * step6_re_dot_v_re
                                              + dstep7_re_dot_v_re * step6_re
                                              - dstep7_re * step6_im_dot_v_im
                                              + dstep7_re_dot_v_im * step6_im)

outer_sum = (T.outer(step6_re_dot_v_re, v_re) 
             + T.outer(step6_re_dot_v_im, v_im)
             - T.outer(step6_im_dot_v_im, v_re)
             + T.outer(step6_im_dot_v_re, v_im))
dstep7_re_RedOpdv_re_term2 = 4. / (vstarv**2) * T.outer((dstep7_re * outer_sum).sum(axis=1), v_re)

dstep7_im_ImdOpdv_re_term1 = - 2. / vstarv * (dstep7_im * step6_im_dot_v_re
                                              + dstep7_im_dot_v_re * step6_im
                                              + dstep7_im * step6_re_dot_v_im
                                              - dstep7_im_dot_v_im * step6_re)

outer_sum = (T.outer(step6_im_dot_v_re, v_re) 
             + T.outer(step6_im_dot_v_im, v_im)
             + T.outer(step6_re_dot_v_im, v_re)
             - T.outer(step6_re_dot_v_re, v_im))
dstep7_im_ImdOpdv_re_term2 = 4. / (vstarv**2) * T.outer((dstep7_im * outer_sum).sum(axis=1), v_re)

dv_re_contribution = (dstep7_re_RedOpdv_re_term1 + dstep7_re_RedOpdv_re_term2 
                      + dstep7_im_ImdOpdv_re_term1 + dstep7_im_ImdOpdv_re_term2)
    
#---------

dstep7_re_RedOpdv_im_term1 = - 2. / vstarv * (dstep7_re * step6_re_dot_v_im
                                              + dstep7_re_dot_v_im * step6_re
                                              - dstep7_re_dot_v_re * step6_im
                                              + dstep7_re * step6_im_dot_v_re)

outer_sum = (T.outer(step6_re_dot_v_re, v_re) 
             + T.outer(step6_re_dot_v_im, v_im)
             - T.outer(step6_im_dot_v_im, v_re)
             + T.outer(step6_im_dot_v_re, v_im))
dstep7_re_RedOpdv_im_term2 = 4. / (vstarv**2) * T.outer((dstep7_re * outer_sum).sum(axis=1), v_im)


dstep7_im_ImdOpdv_im_term1 = - 2. / vstarv * (dstep7_im * step6_im_dot_v_im
                                              + dstep7_im_dot_v_im * step6_im
                                              + dstep7_im_dot_v_re * step6_re
                                              - dstep7_im * step6_re_dot_v_re)

outer_sum = (T.outer(step6_im_dot_v_re, v_re) 
             + T.outer(step6_im_dot_v_im, v_im)
             + T.outer(step6_re_dot_v_im, v_re)
             - T.outer(step6_re_dot_v_re, v_im))
dstep7_im_ImdOpdv_im_term2 = 4. / (vstarv**2) * T.outer((dstep7_im * outer_sum).sum(axis=1), v_im)

dv_im_contribution = (dstep7_re_RedOpdv_im_term1 + dstep7_re_RedOpdv_im_term2 
                      + dstep7_im_ImdOpdv_im_term1 + dstep7_im_ImdOpdv_im_term2)


#---------------------------------------------------------

dv_full = T.concatenate([dv_re_contribution, dv_im_contribution], axis=1)

dv = dv_full.mean(axis=0)

actual_dv = T.grad(c, v)


import pdb; pdb.set_trace()
