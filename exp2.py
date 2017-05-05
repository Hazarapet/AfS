import json
from PIL import Image
import pandas as pd
import numpy as np
import theano.tensor as T
import theano
import sys
from utils.common import f2_score
import keras.backend as K

sys.stdout = open('logging.log', 'w')

print 'hello'
print '{}/{}'.format(42, 158)
