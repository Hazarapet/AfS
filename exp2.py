import sys
import cv2
import json
import numpy as np
import utils.common as common_util
from PIL import Image
import matplotlib.pyplot as plt
import theano.tensor as T

A = np.arange(900).reshape((100, 3, 3))
B = np.arange(100)

for z in common_util.iterate_minibatches(zip(A, B), batchsize=2):
    print z[:, 1].shape
    print np.stack(z[:, 0])
    print z[:, 1]
    sys.exit(0)