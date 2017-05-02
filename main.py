import cv2
import time
import json
import pandas as pd
import numpy as np
from utils.common import f2_score
from keras.optimizers import SGD, Adam
from models.A.model import model as A_model

st_time = time.time()
N_EPOCH = 5
BATCH_SIZE = 80
IMAGE_WIDTH = 128
IMAGE_HEIGH = 128

x_train = []
y_train = []

df_train = pd.read_csv('train.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

print labels

# for f, tags in df_train.values[:20000]:
#     img = cv2.imread('../input/train-jpg/{}.jpg'.format(f))
#     targets = np.zeros(17)
#     for t in tags.split(' '):
#         targets[label_map[t]] = 1
#     x_train.append(cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGH)))
#     y_train.append(targets)

[model, structure] = A_model()

adam = Adam(lr=0.0001, decay=0.)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

print("\n{:.2f}m Runtime".format((time.time() - st_time) / 60))
print "====== End ======"