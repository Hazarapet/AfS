import cv2
import sys
import time
import json
import theano
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from utils import common
from keras import backend as K
from theano.ifelse import ifelse
from keras.models import model_from_json


def model_1_0(out, a):
    return out > a

st_time = time.time()

df_val = pd.read_csv('val_split.csv')
df_train = pd.read_csv('train_v2.csv')

a = T.fvector('a')
x = T.scalar('x')
y = T.scalar('y')
z = T.scalar('z')

f = theano.function([x, y], x*y)

print f(2, 9)
sys.exit()
# 0.92475
weights_path_386_612 = 'models/nm/structures/tr_l:0.1016-tr_f2:0.9143-val_l:0.1081-val_f2:0.9094-time:12-07-2017-03:42:50-dur:386.612.h5'
model_structure_386_612 = 'models/nm/structures/tr_l:0.1016-tr_f2:0.9143-val_l:0.1081-val_f2:0.9094-time:12-07-2017-03:42:50-dur:386.612.json'

with open(model_structure_386_612, 'r') as model_json_386_612:

    # 256x256
    model_386_612 = model_from_json(json.loads(model_json_386_612.read()))
    model_386_612.load_weights(weights_path_386_612)

    print 'model is loaded'

    model_1 = K.function([model_386_612.inputs[0], K.learning_phase(), a], model_1_0(model_386_612.outputs[0], a))

    loss_model = K.function([a], model_1)
    print 'function is compiled'

    for f in common.iterate_minibatches(df_val['image_name'].values[:10], batchsize=1):
        test_batch_inputs = []

        img = cv2.imread('resource/train-jpg/{}.jpg'.format(f[0]))

        img = cv2.resize(img, (256, 256)).astype(np.float32)
        img = img / 256.

        img = img.transpose((2, 0, 1))

        inputs = img

        test_batch_inputs.append(inputs)
        test_batch_inputs = common.aug(test_batch_inputs, inputs)

        print model_1([test_batch_inputs, 0])

print('\n{:.2f}m Runtime'.format((time.time() - st_time) / 60))
print '==== End ===='