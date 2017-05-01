import time
import json
import panda as pn
import numpy as np
from keras.optimizers import SGD, Adam
from utils.common import f2_score

st_time = time.time()
N_EPOCH = 5
BATCH_SIZE = 80


print("{:.2f}m Runtime".format((time.time() - st_time) / 60))
print "====== End ======"