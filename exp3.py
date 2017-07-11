import sys
import cv2
import h5py
import json
import numpy as np
import pandas as pd
from utils import common as common_util
from keras.models import model_from_json


def aug(array, input):
    # rotate 90
    rt90 = np.rot90(input, 1, axes=(1, 2))
    array.append(rt90)

    # flip h
    flip_h = np.flip(input, 2)
    array.append(flip_h)

    # flip v
    flip_v = np.flip(input, 1)
    array.append(flip_v)

    # rotate 90, flip v
    rot90_flip_v = np.rot90(flip_v, 1, axes=(1, 2))
    array.append(rot90_flip_v)

    # rotate 90, flip h
    rot90_flip_h = np.rot90(flip_h, 1, axes=(1, 2))
    array.append(rot90_flip_h)

    return array


df_val = pd.read_csv('val_split.csv')

weights_path = 'models/nm/structures/tr_l:0.0876-tr_f2:0.9138-val_l:0.0785-val_f2:0.9242-time:07-07-2017-21:57:54-dur:473.778.h5'
model_structure = 'models/nm/structures/tr_l:0.0876-tr_f2:0.9138-val_l:0.0785-val_f2:0.9242-time:07-07-2017-21:57:54-dur:473.778.json'

with open(model_structure, 'r') as model_json:
    main_model = model_from_json(json.loads(model_json.read()))
    main_model.load_weights(weights_path)
    print 'model is loaded!'

    # loading the data
    count = 0
    result = []
    print 'images loading...'

    for f in common_util.iterate_minibatches(['train_27881'], batchsize=1):
        test_batch_inputs = []

        img = cv2.imread('resource/train-jpg/{}.jpg'.format(f[0]))

        img = cv2.resize(img, (224, 224)).astype(np.float32)
        img = img.transpose((2, 0, 1))

        inputs = img

        test_batch_inputs.append(inputs)
        test_batch_inputs = aug(test_batch_inputs, inputs)

        test_batch_inputs = np.array(test_batch_inputs).astype(np.float32)

        p_group_test = main_model.predict_on_batch(test_batch_inputs)

        rs = p_group_test[:, [14, 1, 11, 3, 15, 6]]
        print rs, 'mean: ', np.mean(rs, axis=0)
        sys.exit()
        print '{}/{} predicted'.format(count, len(1))






