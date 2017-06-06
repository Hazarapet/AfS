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
HABLOG = ['habitation', 'selective_logging']

def aug(array, input):
    rt90 = np.rot90(input, 1, axes=(1, 2))
    array.append(rt90)

    # rotate 180
    rt180 = np.rot90(input, 2, axes=(1, 2))
    array.append(rt180)

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
    weights_path = 'models/group/structures/tr_l:0.0308-tr_a:1.0-tr_f2:1.0-val_l:0.4826-val_a:0.6725-val_f2:0.6306-time:26-05-2017-02:47:42-dur:344.883.h5'
    model_structure = 'models/group/structures/tr_l:0.0308-tr_a:1.0-tr_f2:1.0-val_l:0.4826-val_a:0.6725-val_f2:0.6306-time:26-05-2017-02:47:42-dur:344.883.json'

    with open(model_structure, 'r') as model_json:
        group_model = model_from_json(json.loads(model_json.read()))
        group_model.load_weights(weights_path)
        print 'group_model is loaded!'

    weights_path = 'models/agriculture/structures/tr_l:0.0045-tr_a:1.0-tr_f2:1.0-val_l:0.6428-val_a:0.7749-val_f2:0.9054-time:04-06-2017-01:43:27-dur:220.152.h5'
    model_structure = 'models/agriculture/structures/tr_l:0.0045-tr_a:1.0-tr_f2:1.0-val_l:0.6428-val_a:0.7749-val_f2:0.9054-time:04-06-2017-01:43:27-dur:220.152.json'

    with open(model_structure, 'r') as model_json:
        agriculture_model = model_from_json(json.loads(model_json.read()))
        agriculture_model.load_weights(weights_path)
        print 'agriculture_model is loaded!'

    weights_path = 'models/burn/structures/tr_l:0.0009-tr_a:1.0-tr_f2:1.0-val_l:3.9634-val_a:0.9831-val_f2:0.6827-time:01-06-2017-22:19:02-dur:152.477.h5'
    model_structure = 'models/burn/structures/tr_l:0.0009-tr_a:1.0-tr_f2:1.0-val_l:3.9634-val_a:0.9831-val_f2:0.6827-time:01-06-2017-22:19:02-dur:152.477.json'

    with open(model_structure, 'r') as model_json:
        burn_model = model_from_json(json.loads(model_json.read()))
        burn_model.load_weights(weights_path)
        print 'burn_model is loaded!'

    weights_path = 'models/clouds/structures/tr_l:0.0047-tr_a:1.0-tr_f2:1.0-val_l:0.2807-val_a:0.902-val_f2:0.8967-time:26-05-2017-19:47:57-dur:217.969.h5'
    model_structure = 'models/clouds/structures/tr_l:0.0047-tr_a:1.0-tr_f2:1.0-val_l:0.2807-val_a:0.902-val_f2:0.8967-time:26-05-2017-19:47:57-dur:217.969.json'

    with open(model_structure, 'r') as model_json:
        clouds_model = model_from_json(json.loads(model_json.read()))
        clouds_model.load_weights(weights_path)
        print 'clouds_model is loaded!'

    weights_path = 'models/hablog/structures/tr_l:0.0694-tr_a:1.0-tr_f2:1.0-val_l:0.4415-val_a:0.8219-val_f2:0.8167-time:03-06-2017-21:53:30-dur:177.196.h5'
    model_structure = 'models/hablog/structures/tr_l:0.0694-tr_a:1.0-tr_f2:1.0-val_l:0.4415-val_a:0.8219-val_f2:0.8167-time:03-06-2017-21:53:30-dur:177.196.json'

    with open(model_structure, 'r') as model_json:
        hablog_model = model_from_json(json.loads(model_json.read()))
        hablog_model.load_weights(weights_path)
        print 'hablog_model is loaded!'

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
        p_agr_test = agg(p_agr_test) # avg of prediction

        prediction_vector[label_map['agriculture']] = p_agr_test

        # -------------------------------------------- #
        # ----------------- Burn --------------------- #
        test_batch_inputs = []

        # red, green, blue, ndwi, ndvi, ior, gemi
        inputs = [red, green, blue, nir, ndvi, ior, bai, gemi]

        test_batch_inputs.append(inputs)
        test_batch_inputs = aug(test_batch_inputs, inputs)
        test_batch_inputs = np.array(test_batch_inputs).astype(np.float32)

        p_burn_test = burn_model.predict_on_batch(test_batch_inputs)
        p_burn_test = agg(p_burn_test) # avg of prediction

        prediction_vector[label_map['slash_burn']] = p_burn_test

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
        # ----------------- Hablog ------------------- #
        test_batch_inputs = []

        # red, green, blue, ndwi, ndvi, ior, gemi
        inputs = [red, green, blue, nir, ndvi, ior, bai, gemi]

        test_batch_inputs.append(inputs)
        test_batch_inputs = aug(test_batch_inputs, inputs)
        test_batch_inputs = np.array(test_batch_inputs).astype(np.float32)

        p_hablog_test = hablog_model.predict_on_batch(test_batch_inputs)
        p_hablog_test = agg(p_hablog_test)  # avg of prediction

        for l, p in zip(HABLOG, p_hablog_test):
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


