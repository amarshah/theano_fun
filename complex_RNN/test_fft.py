import theano
from fftconv import cufft, cuifft
import numpy as np
import theano.tensor as T


x = T.tensor3()
trans = cufft(x) / T.sqrt(x.shape[1])
y = T.grad(T.mean(trans), x)

cost = theano.function([x], y)

asd = cost(np.asarray(np.random.rand(10,9,2), dtype='float32'))

import pdb; pdb.set_trace()













