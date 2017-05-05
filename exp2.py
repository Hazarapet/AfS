from PIL import Image
import pandas as pd
import numpy as np
import theano.tensor as T
import theano
import keras.backend as K

X = T.matrix('x')
B = X * [1, 1, 0]
D = B > .5
D = D * .5
H = X - D

f = theano.function([X], H)
r = np.random.random((2, 3)).astype(np.float16)

print r
print f(r)
