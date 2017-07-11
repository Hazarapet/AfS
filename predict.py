import os
import sys
import cv2
import json
import numpy as np
import pandas as pd
from utils import common as common_util
from utils import image as UtilImage
from keras.models import model_from_json

BATCH_SIZE = 100
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256


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


def agg(array):
    return np.mean(array, axis=0)


def result_single_tif(X, path, do_agg=True):
    weights_path = 'models/main/structures/tr_l:0.1412-tr_a:0.9478-tr_f2:0.8678-val_l:0.1472-val_a:0.946-val_f2:0.8632-time:23-06-2017-21:40:17-dur:99.825.h5'
    model_structure = 'models/main/structures/tr_l:0.1412-tr_a:0.9478-tr_f2:0.8678-val_l:0.1472-val_a:0.946-val_f2:0.8632-time:23-06-2017-21:40:17-dur:99.825.json'

    with open(model_structure, 'r') as model_json:
        main_model = model_from_json(json.loads(model_json.read()))
        main_model.load_weights(weights_path)
        print 'model is loaded!'

        # loading the data
        count = 0
        result = []
        print 'images loading...'

        for f in common_util.iterate_minibatches(X, batchsize=1):
            test_batch_inputs = []

            # for f in files:
            rgbn = UtilImage.process_tif(path.format(f[0]))

            ndvi = UtilImage.ndvi(rgbn)
            ndwi = UtilImage.ndwi(rgbn)
            ior = UtilImage.ior(rgbn)
            bai = UtilImage.bai(rgbn)

            # resize
            red = cv2.resize(rgbn[0], (IMAGE_WIDTH, IMAGE_HEIGHT))
            green = cv2.resize(rgbn[1], (IMAGE_WIDTH, IMAGE_HEIGHT))
            blue = cv2.resize(rgbn[2], (IMAGE_WIDTH, IMAGE_HEIGHT))
            nir = cv2.resize(rgbn[3], (IMAGE_WIDTH, IMAGE_HEIGHT))
            ndvi = cv2.resize(ndvi, (IMAGE_WIDTH, IMAGE_HEIGHT))
            ndwi = cv2.resize(ndwi, (IMAGE_WIDTH, IMAGE_HEIGHT))
            ior = cv2.resize(ior, (IMAGE_WIDTH, IMAGE_HEIGHT))
            bai = cv2.resize(bai, (IMAGE_WIDTH, IMAGE_HEIGHT))

            # red, green, blue, nir, ndvi, ndwi, ior, bai
            inputs = [red, green, blue, nir, ndvi, ndwi, ior, bai]
            test_batch_inputs.append(inputs)
            test_batch_inputs = aug(test_batch_inputs, np.array(inputs))

            test_batch_inputs = np.array(test_batch_inputs).astype(np.float32)

            p_group_test = main_model.predict_on_batch(test_batch_inputs)

            count += 1
            if do_agg:
                result.append(agg(p_group_test))
            else:
                result = result + p_group_test.tolist()

            print '{}/{} predicted'.format(count, len(X))

    return result


def result_single_jpg(X, path, weights_path, model_structure, do_agg=True):

    with open(model_structure, 'r') as model_json:
        main_model = model_from_json(json.loads(model_json.read()))
        main_model.load_weights(weights_path)
        print 'model is loaded!'

        # loading the data
        count = 0
        result = []
        print 'images loading...'

        for f in common_util.iterate_minibatches(X, batchsize=1):
            test_batch_inputs = []

            img = cv2.imread(path.format(f[0]))

            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.float32)
            img = img.transpose((2, 0, 1))

            inputs = img

            test_batch_inputs.append(inputs)
            test_batch_inputs = aug(test_batch_inputs, inputs)

            test_batch_inputs = np.array(test_batch_inputs).astype(np.float32)

            p_group_test = main_model.predict_on_batch(test_batch_inputs)

            count += 1
            if do_agg:
                result.append(agg(p_group_test))
            else:
                result = result + p_group_test.tolist()

            print '{}/{} predicted'.format(count, len(X))

    return result

