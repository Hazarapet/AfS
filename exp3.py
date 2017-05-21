import sys
import numpy as np
from utils import common as common_util

A = np.arange(27).reshape((3, 3, 3))
B = np.arange(12).reshape((3, 4))

for min_b in common_util.iterate_minibatches(zip(A, B), batchsize=2):
    print np.stack(min_b[:, 1])

