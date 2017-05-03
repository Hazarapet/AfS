import json
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from keras.models import model_from_json

IMAGE_WIDTH = 128
IMAGE_HEIGH = 128

file_p = 'resource/test128x128/26547074743__flickr.jpg'
weights_path = 'models/A/structures/tr_l:0.221-tr_a:0.938-val_l:0.214-val_a:0.933-time:03-05-2017-16:37:57-dur:14.568.h5'
model_structure = 'models/A/structures/tr_l:0.221-tr_a:0.938-val_l:0.214-val_a:0.933-time:03-05-2017-16:37:57-dur:14.568.json'

with open(model_structure, 'r') as model_json:
    model = model_from_json(json.loads(model_json.read()))
    model.load_weights(weights_path)


print 'train.csv loading...'
# loading the data
df_train = pd.read_csv('train.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

# we should shuffle all examples
np.random.shuffle(df_train.values)

# splitting to train and validation set
index = int(len(df_train.values) * 0.8)
train, val = df_train.values[:index], df_train.values[index:]

v_batch_inputs = []
v_batch_labels = []

print 'images loading...'
# load val's images
for f, tags in val:
    img = cv2.imread('resource/train-jpg/{}.jpg'.format(f))
    assert img is not None

    if img is not None:
        targets = np.zeros(17)
        for t in tags.split(' '):
            targets[label_map[t]] = 1

        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGH)).astype(np.float16)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img.transpose((2, 0, 1))

        v_batch_inputs.append(img)
        v_batch_labels.append(targets)

v_batch_inputs = np.array(v_batch_inputs).astype(np.float16)
v_batch_labels = np.array(v_batch_labels).astype(np.int8)

print 'predicting...'
p_valid = model.predict(v_batch_inputs, batch_size=80)

print 'some results:'
print v_batch_labels[:10], p_valid[:10]

print 'F2_Score', fbeta_score(v_batch_labels, np.array(p_valid) > 0.5, beta=2, average='samples')