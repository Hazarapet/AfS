import os
import sys
import cv2
import json
import numpy as np
import pandas as pd
from utils import common as common_util
from utils import image as UtilImage
from keras.models import model_from_json

BATCH_SIZE = 1
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


def result(X, path):
    weights_path = 'models/group/structures/tr_l:0.0173-tr_a:1.0-tr_f2:1.0-val_l:0.6767-val_a:0.648-val_f2:0.5519-time:08-06-2017-22:03:40-dur:386.923.h5'
    model_structure = 'models/group/structures/tr_l:0.0173-tr_a:1.0-tr_f2:1.0-val_l:0.6767-val_a:0.648-val_f2:0.5519-time:08-06-2017-22:03:40-dur:386.923.json'

    with open(model_structure, 'r') as model_json:
        group_model = model_from_json(json.loads(model_json.read()))
        group_model.load_weights(weights_path)
        print 'group_model is loaded!'

    weights_path = 'models/agriculture/structures/tr_l:0.0494-tr_a:1.0-tr_f2:1.0-val_l:0.5505-val_a:0.8312-val_f2:0.8868-time:09-06-2017-15:38:26-dur:186.136.h5'
    model_structure = 'models/agriculture/structures/tr_l:0.0494-tr_a:1.0-tr_f2:1.0-val_l:0.5505-val_a:0.8312-val_f2:0.8868-time:09-06-2017-15:38:26-dur:186.136.json'

    with open(model_structure, 'r') as model_json:
        agriculture_model = model_from_json(json.loads(model_json.read()))
        agriculture_model.load_weights(weights_path)
        print 'agriculture_model is loaded!'

    weights_path = 'models/clouds/structures/tr_l:0.0047-tr_a:1.0-tr_f2:1.0-val_l:0.2807-val_a:0.902-val_f2:0.8967-time:26-05-2017-19:47:57-dur:217.969.h5'
    model_structure = 'models/clouds/structures/tr_l:0.0047-tr_a:1.0-tr_f2:1.0-val_l:0.2807-val_a:0.902-val_f2:0.8967-time:26-05-2017-19:47:57-dur:217.969.json'

    with open(model_structure, 'r') as model_json:
        clouds_model = model_from_json(json.loads(model_json.read()))
        clouds_model.load_weights(weights_path)
        print 'clouds_model is loaded!'

    weights_path = 'models/small_group/structures/tr_l:0.0749-tr_a:1.0-tr_f2:1.0-val_l:0.3815-val_a:0.6151-val_f2:0.8419-time:09-06-2017-19:51:05-dur:247.011.h5'
    model_structure = 'models/small_group/structures/tr_l:0.0749-tr_a:1.0-tr_f2:1.0-val_l:0.3815-val_a:0.6151-val_f2:0.8419-time:09-06-2017-19:51:05-dur:247.011.json'

    with open(model_structure, 'r') as model_json:
        small_group_model = model_from_json(json.loads(model_json.read()))
        small_group_model.load_weights(weights_path)
        print 'small_group_model is loaded!'

    weights_path = 'models/primary/structures/tr_l:0.0035-tr_a:1.0-tr_f2:1.0-val_l:0.4134-val_a:0.862-val_f2:0.946-time:24-05-2017-18:30:15-dur:53.228.h5'
    model_structure = 'models/primary/structures/tr_l:0.0035-tr_a:1.0-tr_f2:1.0-val_l:0.4134-val_a:0.862-val_f2:0.946-time:24-05-2017-18:30:15-dur:53.228.json'

    with open(model_structure, 'r') as model_json:
        primary_model = model_from_json(json.loads(model_json.read()))
        primary_model.load_weights(weights_path)
        print 'primary_model is loaded!'

    weights_path = 'models/road/structures/tr_l:0.0004-tr_a:1.0-tr_f2:1.0-val_l:0.6021-val_a:0.8187-val_f2:0.8717-time:27-05-2017-05:10:54-dur:231.052.h5'
    model_structure = 'models/road/structures/tr_l:0.0004-tr_a:1.0-tr_f2:1.0-val_l:0.6021-val_a:0.8187-val_f2:0.8717-time:27-05-2017-05:10:54-dur:231.052.json'

    with open(model_structure, 'r') as model_json:
        road_model = model_from_json(json.loads(model_json.read()))
        road_model.load_weights(weights_path)
        print 'road_model is loaded!'

    weights_path = 'models/water/structures/tr_l:0.0119-tr_a:1.0-tr_f2:1.0-val_l:0.4473-val_a:0.8639-val_f2:0.8652-time:27-05-2017-19:20:46-dur:170.095.h5'
    model_structure = 'models/water/structures/tr_l:0.0119-tr_a:1.0-tr_f2:1.0-val_l:0.4473-val_a:0.8639-val_f2:0.8652-time:27-05-2017-19:20:46-dur:170.095.json'

    with open(model_structure, 'r') as model_json:
        water_model = model_from_json(json.loads(model_json.read()))
        water_model.load_weights(weights_path)
        print 'water_model is loaded!'

    print 'train.csv loading...'

    # loading the data
    df_train = pd.read_csv('train_v2.csv')

    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

    label_map = {l: i for i, l in enumerate(labels)}

    count = 0
    result = []
    print 'images loading...'

    for f in common_util.iterate_minibatches(X, batchsize=BATCH_SIZE):
        prediction_vector = np.zeros(17)

        test_batch_inputs = []

        rgbn = UtilImage.process_tif(path.format(f[0]))

        ndvi = UtilImage.ndvi(rgbn)
        ndwi = UtilImage.ndwi(rgbn)
        ior = UtilImage.ior(rgbn)
        bai = UtilImage.bai(rgbn)
        gemi = UtilImage.gemi(rgbn)

        # resize
        red = cv2.resize(rgbn[0], (IMAGE_WIDTH, IMAGE_HEIGHT))
        green = cv2.resize(rgbn[1], (IMAGE_WIDTH, IMAGE_HEIGHT))
        blue = cv2.resize(rgbn[2], (IMAGE_WIDTH, IMAGE_HEIGHT))
        nir = cv2.resize(rgbn[3], (IMAGE_WIDTH, IMAGE_HEIGHT))
        ndvi = cv2.resize(ndvi, (IMAGE_WIDTH, IMAGE_HEIGHT))
        ndwi = cv2.resize(ndwi, (IMAGE_WIDTH, IMAGE_HEIGHT))
        ior = cv2.resize(ior, (IMAGE_WIDTH, IMAGE_HEIGHT))
        bai = cv2.resize(bai, (IMAGE_WIDTH, IMAGE_HEIGHT))
        gemi = cv2.resize(gemi, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # ----------------------------------------------------------------------------
        # ---------------------------------- Models ----------------------------------
        # ----------------------------------------------------------------------------
        # red, green, blue, nir, ndvi, ndwi, ior, bai, gemi, grvi, vari, gndvi, sr, savi, lai
        inputs = [red, green, blue, nir, ndvi, ndwi, ior, bai]

        test_batch_inputs.append(inputs)
        test_batch_inputs = aug(test_batch_inputs, inputs)
        test_batch_inputs = np.array(test_batch_inputs).astype(np.float32)

        p_group_test = group_model.predict_on_batch(test_batch_inputs)
        p_group_test = agg(p_group_test)

        for l, p in zip(GROUP, p_group_test):
            prediction_vector[label_map[l]] = p

        # -------------------------------------------- #
        # --------------- Agriculture ---------------- #
        test_batch_inputs = []

        # red, green, blue, ndwi, ndvi, ior, gemi
        inputs = [red, green, blue, ndvi, ndwi, ior, gemi]

        test_batch_inputs.append(inputs)
        test_batch_inputs = aug(test_batch_inputs, inputs)
        test_batch_inputs = np.array(test_batch_inputs).astype(np.float32)

        p_agr_test = agriculture_model.predict_on_batch(test_batch_inputs)
        p_agr_test = agg(p_agr_test)  # avg of prediction

        prediction_vector[label_map['agriculture']] = p_agr_test

        # -------------------------------------------- #
        # ------------------ Clouds ------------------ #
        test_batch_inputs = []

        # red, green, blue, ndwi, ndvi, ior, gemi
        inputs = [red, green]

        test_batch_inputs.append(inputs)
        test_batch_inputs = aug(test_batch_inputs, inputs)
        test_batch_inputs = np.array(test_batch_inputs).astype(np.float32)

        p_clouds_test = clouds_model.predict_on_batch(test_batch_inputs)
        p_clouds_test = agg(p_clouds_test)  # avg of prediction

        for l, p in zip(CLOUDS, p_clouds_test):
            prediction_vector[label_map[l]] = p

        # -------------------------------------------- #
        # ----------------- Small Group ------------------- #
        test_batch_inputs = []

        # red, green, blue, ndwi, ndvi, ior, gemi
        inputs = [red, green, blue, nir, ndvi, ior, bai, gemi]

        test_batch_inputs.append(inputs)
        test_batch_inputs = aug(test_batch_inputs, inputs)
        test_batch_inputs = np.array(test_batch_inputs).astype(np.float32)

        p_small_group_test = small_group_model.predict_on_batch(test_batch_inputs)
        p_small_group_test = agg(p_small_group_test)  # avg of prediction

        for l, p in zip(SMALL_GROUP, p_small_group_test):
            prediction_vector[label_map[l]] = p

        # -------------------------------------------- #
        # ----------------- Primary ------------------ #
        test_batch_inputs = []

        # red, green, blue
        inputs = [red, green, blue]

        test_batch_inputs.append(inputs)
        test_batch_inputs = aug(test_batch_inputs, inputs)
        test_batch_inputs = np.array(test_batch_inputs).astype(np.float32)

        p_primary_test = primary_model.predict_on_batch(test_batch_inputs)
        p_primary_test = agg(p_primary_test) # avg of prediction

        prediction_vector[label_map['primary']] = p_primary_test

        # -------------------------------------------- #
        # ------------------- Road ------------------- #
        test_batch_inputs = []

        # TODO fix input
        # red, green, blue, nir, ndvi, ior, bai
        inputs = [red, green, blue, nir, ndvi, ior, bai]

        test_batch_inputs.append(inputs)
        test_batch_inputs = aug(test_batch_inputs, inputs)
        test_batch_inputs = np.array(test_batch_inputs).astype(np.float32)

        p_road_test = road_model.predict_on_batch(test_batch_inputs)
        p_road_test = agg(p_road_test)  # avg of prediction

        prediction_vector[label_map['road']] = p_road_test

        # -------------------------------------------- #
        # ------------------ Water ------------------- #
        test_batch_inputs = []

        # ior, ndvi, ndwi
        inputs = [ior, ndvi, ndwi]

        test_batch_inputs.append(inputs)

        test_batch_inputs = aug(test_batch_inputs, inputs)

        test_batch_inputs = np.array(test_batch_inputs).astype(np.float32)

        p_road_test = water_model.predict_on_batch(test_batch_inputs)
        p_road_test = agg(p_road_test)  # avg of prediction

        prediction_vector[label_map['water']] = p_road_test

        count += BATCH_SIZE
        result.append(prediction_vector)

        print '{}/{} predicted'.format(count, len(X))

    return result


def result_single_jpg(X, path):
    weights_path = 'models/nm/structures/tr_l:0.1666-tr_a:0.2941-tr_f2:0.8559-val_l:0.2524-val_a:0.4652-val_f2:0.818-time:20-06-2017-10:28:19-dur:569.032.h5'
    model_structure = 'models/nm/structures/tr_l:0.1666-tr_a:0.2941-tr_f2:0.8559-val_l:0.2524-val_a:0.4652-val_f2:0.818-time:20-06-2017-10:28:19-dur:569.032.json'

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

            img = cv2.imread(path.format(f[0]))

            img = cv2.resize(img, (224, 224)).astype(np.float16)
            img[:, :, 0] -= 103.939
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 123.68
            img = img.transpose((2, 0, 1))

            inputs = img

            test_batch_inputs.append(inputs)
            test_batch_inputs = aug(test_batch_inputs, inputs)
            test_batch_inputs = np.array(test_batch_inputs).astype(np.float32)

            p_group_test = main_model.predict_on_batch(test_batch_inputs)
            p_group_test = agg(p_group_test)

            count += BATCH_SIZE
            result.append(p_group_test)

            print '{}/{} predicted'.format(count, len(X))

    return result
