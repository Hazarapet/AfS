import os
import cv2
import numpy as np
import theano.tensor as T
from theano import function

x, y = T.dscalars('x', 'y', dtype='int8')
z = x + y

f = function([x, y], z)

print f(2, 3)



