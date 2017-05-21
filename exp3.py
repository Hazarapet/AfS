import sys
import numpy as np
import keras.backend as K
from utils import common as common_util

A = np.random.random((3, 3))
p = np.random.random((3, 3))
A = (A > 0.5) * 1.
p = (p > 0.5) * 1.
print A
print p

print '------------------------'
print '------------------------'
print common_util.f2_score(A, p).eval()

