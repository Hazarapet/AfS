import os
import cv2
import json
import numpy as np
import pandas as pd
from utils import common as common_util
from utils import image as UtilImage
from sklearn.metrics import fbeta_score
from keras.models import model_from_json

BATCH_SIZE = 100
IMAGE_WIDTH = 128
IMAGE_HEIGH = 128

weights_path = 'models/UNET/structures/tr_l:0.049-tr_a:0.987-val_l:0.116-val_a:0.96-time:11-05-2017-02:39:02-dur:640.645.h5'
model_structure = 'models/UNET/structures/tr_l:0.049-tr_a:0.987-val_l:0.116-val_a:0.96-time:11-05-2017-02:39:02-dur:640.645.json'

with open(model_structure, 'r') as model_json:
    model = model_from_json(json.loads(model_json.read()))
    model.load_weights(weights_path)


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
X_test = os.listdir('resource/test-jpg')

for min_batch in common_util.iterate_minibatches(X_test, batchsize=BATCH_SIZE):
    test_batch_inputs = []
    # test_batch_sobel_inputs = []

    # load val's images
    for f in min_batch:
        img = cv2.imread('resource/test-jpg/{}'.format(f))
        assert img is not None

        if img is not None:
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGH)).astype(np.float32)
            img[:, :, 0] -= 103.939
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 123.68
            img = img.transpose((2, 0, 1))

            # mag, angle = UtilImage.img_sobel(img)

            test_batch_inputs.append(img)
            # test_batch_sobel_inputs.append(mag)

    count += len(test_batch_inputs)
    test_batch_inputs = np.array(test_batch_inputs).astype(np.float16)
    # test_batch_sobel_inputs = np.array(test_batch_sobel_inputs).astype(np.float16)

    p_test = model.predict_on_batch(test_batch_inputs)
    result.extend(p_test)
    files.extend(min_batch)

    print '{}/{} predicted'.format(count, len(X_test))

thres = [0.1, 0.23, 0.04, 0.22, 0.16, 0.2, 0.26, 0.24, 0.23, 0.14, 0.33, 0.19, 0.17, 0.07, 0.25, 0.24, 0.12]
threz = [0.22, 0.3, 0.24, 0.33, 0.24, 0.24, 0.26, 0.25, 0.26, 0.24, 0.25, 0.24, 0.24, 0.24, 0.3, 0.25, 0.24]

df_test = pd.DataFrame([[p.replace('.jpg', ''), p] for p in X_test])
df_test.columns = ['image_name', 'tags']

tags = []
for r in result:
    r = list(r > .19)
    t = [inv_label_map[i] for i, j in enumerate(r) if j]
    tags.append(' '.join(t))

df_test['tags'] = tags
df_test.to_csv('submission_0.csv', index=False)

