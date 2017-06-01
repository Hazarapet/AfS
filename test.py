import os
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

weights_path = 'models/UNET/structures/tr_l:0.04-tr_a:0.99-val_l:0.076-val_a:0.975-time:11-05-2017-19:25:31-dur:227.379.h5'
model_structure = 'models/UNET/structures/tr_l:0.04-tr_a:0.99-val_l:0.076-val_a:0.975-time:11-05-2017-19:25:31-dur:227.379.json'

with open(model_structure, 'r') as model_json:
    group_model = model_from_json(json.loads(model_json.read()))
    group_model.load_weights(weights_path)

weights_path = 'models/UNET/structures/tr_l:0.04-tr_a:0.99-val_l:0.076-val_a:0.975-time:11-05-2017-19:25:31-dur:227.379.h5'
model_structure = 'models/UNET/structures/tr_l:0.04-tr_a:0.99-val_l:0.076-val_a:0.975-time:11-05-2017-19:25:31-dur:227.379.json'

with open(model_structure, 'r') as model_json:
    agriculture_model = model_from_json(json.loads(model_json.read()))
    agriculture_model.load_weights(weights_path)


print 'train.csv loading...'
# loading the data
df_train = pd.read_csv('train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

count = 0
result = []
files = []
print 'images loading...'
X_test = os.listdir('resource/test-v2-tif')

for f in common_util.iterate_minibatches(X_test, batchsize=1):
    test_batch_inputs = []

    rgbn = UtilImage.process_tif('resource/train-tif-v2/{}.tif'.format(f))
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

    # red, green, blue, nir, ndvi, ndwi, ior, bai, gemi, grvi, vari, gndvi, sr, savi, lai
    inputs = [red, green, blue, nir, ndvi, ndwi, ior, bai]

    test_batch_inputs.append(inputs)

    test_batch_inputs = aug(test_batch_inputs, inputs)

    count += len(test_batch_inputs)
    test_batch_inputs = np.array(test_batch_inputs).astype(np.float32)

    p_group_test = group_model.predict_on_batch(test_batch_inputs)
    p_group_test = np.sum(p_group_test, axis=0) / 5

    result.extend(p_group_test)

    # -------------------------------------------- #
    test_batch_inputs = []

    # red, green, blue, ndwi, ndvi, ior, gemi
    inputs = [red, green, blue, ndvi, ndwi, ior, gemi]

    test_batch_inputs.append(inputs)

    test_batch_inputs = aug(test_batch_inputs, inputs)

    count += len(test_batch_inputs)
    test_batch_inputs = np.array(test_batch_inputs).astype(np.float32)

    p_agr_test = agriculture_model.predict_on_batch(test_batch_inputs)
    p_agr_test = np.sum(p_agr_test, axis=0) / 5

    result.extend(p_agr_test)

    print '{}/{} predicted'.format(count, len(X_test))

thres = [0.085, 0.2375, 0.19, 0.2625, 0.16, 0.0875, 0.205, 0.1925, 0.265, 0.1625, 0.1375, 0.2175, 0.2225, 0.0475, 0.245, 0.21, 0.14] # Heng CherKeng's example
threz = [0.22, 0.3, 0.24, 0.33, 0.24, 0.24, 0.26, 0.25, 0.26, 0.24, 0.25, 0.24, 0.24, 0.24, 0.3, 0.25, 0.24]

df_test = pd.DataFrame([[p.replace('.jpg', ''), p] for p in X_test])
df_test.columns = ['image_name', 'tags']

tags = []
for r in result:
    r = list(r > .2)
    t = [inv_label_map[i] for i, j in enumerate(r) if j]
    tags.append(' '.join(t))

df_test['tags'] = tags
df_test.to_csv('submission_0.csv', index=False)

