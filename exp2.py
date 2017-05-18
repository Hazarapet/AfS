import cv2
import json
import numpy as np
import utils.common as common_util
from PIL import Image
import matplotlib.pyplot as plt

A = np.arange(360).reshape((10, 4, 3, 3))
B = np.arange(10)

for t_b, t_l in common_util.iterate_minibatches(zip(A, B), batchsize=40):
    print len(t_b)