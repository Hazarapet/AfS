import os
import cv2
import json
import numpy as np
import pandas as pd
from utils import common as common_util
from utils import image as UtilImage
from sklearn.metrics import fbeta_score
from keras.models import model_from_json
from models.water.model import model as water_model

BATCH_SIZE = 100
IMAGE_WIDTH = 128
IMAGE_HEIGH = 128

weights_path = 'models/water/structures/tr_l:0.7265-tr_a:0.6667-tr_f2:0.8453-val_l:0.7318-val_a:0.1784-val_f2:0.5153-time:19-05-2017-16:05:52-dur:9.645.h5'
model_structure = 'models/water/structures/tr_l:0.7265-tr_a:0.6667-tr_f2:0.8453-val_l:0.7318-val_a:0.1784-val_f2:0.5153-time:19-05-2017-16:05:52-dur:9.645.json'
tif_sample = 'train_10010'

with open(model_structure, 'r') as model_json:
    model = model_from_json(json.loads(model_json.read()))
    model.load_weights(weights_path)


# [model, _] = water_model()

print 'train.csv loading...'
# loading the data
df_train = pd.read_csv('train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

index = int(len(df_train.values) * 0.8)
train, val = df_train.values[:index], df_train.values[index:]

rgbn, ndwi, _, _, _ = UtilImage.process_tif('resource/train-tif-v2/{}.tif'.format(tif_sample))

r = cv2.resize(rgbn[0].astype(np.float32), (IMAGE_WIDTH, IMAGE_HEIGH))
g = cv2.resize(rgbn[1].astype(np.float32), (IMAGE_WIDTH, IMAGE_HEIGH))
b = cv2.resize(rgbn[2].astype(np.float32), (IMAGE_WIDTH, IMAGE_HEIGH))
ndwi = cv2.resize(ndwi.astype(np.float32), (IMAGE_WIDTH, IMAGE_HEIGH))

inputs = [r, g, b, ndwi]

inputs = np.array([inputs]).astype(np.float16)

pred = model.predict(inputs)

print pred

