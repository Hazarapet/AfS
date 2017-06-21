import os
import sys
import cv2
import json
import numpy as np
import pandas as pd
from utils import common as common_util
from utils import image as UtilImage
from keras.models import model_from_json

BATCH_SIZE = 200
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

GROUP = ['artisinal_mine',
         'bare_ground',
         'blooming',
         'blow_down',
         'conventional_mine',
         'cultivation',
         'haze',
         'selective_logging']

CLOUDS = ['cloudy', 'partly_cloudy']
SMALL_GROUP = ['habitation', 'clear', 'slash_burn']


def ensemble(array):
    new_array = []
    for cl in range(array.shape[1]):
        cn = list(array[:, cl]).count(1)
        all_cn = array.shape[0]
        if cn > all_cn / 2:
            new_array.append(1)
        else:
            new_array.append(0)
    return new_array


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


def result_single_tif(X, path):
    weights_path = 'models/main/structures/tr_l:0.0608-tr_a:1.0-tr_f2:1.0-val_l:0.3504-val_a:0.3593-val_f2:0.7939-time:19-06-2017-06:44:56-dur:483.84.h5'
    model_structure = 'models/main/structures/tr_l:0.0608-tr_a:1.0-tr_f2:1.0-val_l:0.3504-val_a:0.3593-val_f2:0.7939-time:19-06-2017-06:44:56-dur:483.84.json'

    with open(model_structure, 'r') as model_json:
        main_model = model_from_json(json.loads(model_json.read()))
        main_model.load_weights(weights_path)
        print 'main_model is loaded!'

        # loading the data
        count = 0
        result = []
        print 'images loading...'

        for f in common_util.iterate_minibatches(X, batchsize=BATCH_SIZE):
            test_batch_inputs = []

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
            p_group_test = ensemble(p_group_test)

            count += BATCH_SIZE
            result.append(p_group_test)

            print '{}/{} predicted'.format(count, len(X))

    return result


def result_single_jpg(X, path):
    weights_path = 'models/nm/structures/tr_l:0.1467-tr_a:0.9411-tr_f2:0.7909-val_l:0.2166-val_a:0.9319-val_f2:0.7635-time:20-06-2017-23:28:27-dur:406.82.h5'
    model_structure = 'models/nm/structures/tr_l:0.1467-tr_a:0.9411-tr_f2:0.7909-val_l:0.2166-val_a:0.9319-val_f2:0.7635-time:20-06-2017-23:28:27-dur:406.82.json'

    with open(model_structure, 'r') as model_json:
        main_model = model_from_json(json.loads(model_json.read()))
        main_model.load_weights(weights_path)
        print 'main_model is loaded!'

        # loading the data
        count = 0
        result = []
        print 'images loading...'

        for files in common_util.iterate_minibatches(X, batchsize=BATCH_SIZE):
            test_batch_inputs = []

            for f in files:
                img = cv2.imread(path.format(f))

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

            p_group = []
            for example in np.split(p_group_test, 3):
                p_group.append(ensemble(example))

            count += BATCH_SIZE
            result = result + p_group

            print '{}/{} predicted'.format(count, len(X))

    return result

