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
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

def aug(array, input):
    rt90 = np.rot90(input, 1, axes=(1, 2))
    array.append(rt90)

    # flip h
    flip_h = np.flip(input, 2)
    array.append(flip_h)

    # flip v
    flip_v = np.flip(input, 1)
    array.append(flip_v)

    return array


def agg(array):
    return np.mean(array, axis=0)


def result_single_tif(X, path, do_agg=True):
    weights_path = 'models/main/structures/tr_l:0.1456-tr_a:0.9445-tr_f2:0.8612-val_l:0.1469-val_a:0.9446-val_f2:0.8574-time:22-06-2017-22:41:25-dur:335.64.h5'
    model_structure = 'models/main/structures/tr_l:0.1456-tr_a:0.9445-tr_f2:0.8612-val_l:0.1469-val_a:0.9446-val_f2:0.8574-time:22-06-2017-22:41:25-dur:335.64.json'

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


def result_single_jpg(X, path, do_agg=True):
    weights_path = 'models/nm/structures/tr_l:0.1467-tr_a:0.9411-tr_f2:0.7909-val_l:0.2166-val_a:0.9319-val_f2:0.7635-time:20-06-2017-23:28:27-dur:406.82.h5'
    model_structure = 'models/nm/structures/tr_l:0.1467-tr_a:0.9411-tr_f2:0.7909-val_l:0.2166-val_a:0.9319-val_f2:0.7635-time:20-06-2017-23:28:27-dur:406.82.json'

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

            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.float16)
            img[:, :, 0] -= 103.939
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 123.68
            img = img.transpose((2, 0, 1))

            inputs = img

            test_batch_inputs.append(inputs)
            test_batch_inputs = aug(test_batch_inputs, inputs)

            test_batch_inputs = np.array(test_batch_inputs).astype(np.float16)

            p_group_test = main_model.predict_on_batch(test_batch_inputs)

            count += 1
            if do_agg:
                result.append(agg(p_group_test))
            else:
                result = result + p_group_test.tolist()

            print '{}/{} predicted'.format(count, len(X))

    return result

