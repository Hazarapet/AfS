import os
import sys
import cv2
import time
import json
import numpy as np
import pandas as pd
import predict
import utils.common as common
from keras.models import model_from_json

st_time = time.time()

# loading the data
df_train = pd.read_csv('train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

print 'images loading...'
X_test = os.listdir('resource/test-jpg')

thres_345_134 = [0.27, 0.21, 0.56, 0.08, 0.43, 0.62, 0.3, 0.59, 0.38, 0.08, 0.19, 0.19, 0.44, 0.33, 0.16, 0.19, 0.51]
thres_459_14 = [0.19, 0.09, 0.67, 0.29, 0.55, 0.91, 0.16, 0.4, 0.32, 0.1, 0.21, 0.14, 0.28, 0.17, 0.13, 0.11, 0.18]
thres_600_446 = [0.14, 0.15, 0.14, 0.27, 0.1, 0.35, 0.17, 0.32, 0.24, 0.12, 0.08, 0.1, 0.12, 0.13, 0.18, 0.16, 0.31]

df_test = pd.DataFrame([[p.replace('.jpg', ''), p] for p in X_test])
df_test.columns = ['image_name', 'tags']

weights_path_459_14 = 'models/nm/structures/tr_l:0.1162-tr_f2:0.9084-val_l:0.133-val_f2:0.8983-time:05-07-2017-00:25:12-dur:459.14.h5'
model_structure_459_14 = 'models/nm/structures/tr_l:0.1162-tr_f2:0.9084-val_l:0.133-val_f2:0.8983-time:05-07-2017-00:25:12-dur:459.14.json'

weights_path_345_134 = 'models/nm/structures/tr_l:0.1339-tr_f2:0.8899-val_l:0.1387-val_f2:0.8868-time:04-07-2017-13:50:07-dur:345.134.h5'
model_structure_345_134 = 'models/nm/structures/tr_l:0.1339-tr_f2:0.8899-val_l:0.1387-val_f2:0.8868-time:04-07-2017-13:50:07-dur:345.134.json'

weights_path_600_446 = 'models/nm/structures/tr_l:0.0912-tr_f2:0.9101-val_l:0.0969-val_f2:0.9071-time:07-07-2017-07:43:25-dur:600.446.h5'
model_structure_600_446 = 'models/nm/structures/tr_l:0.0912-tr_f2:0.9101-val_l:0.0969-val_f2:0.9071-time:07-07-2017-07:43:25-dur:600.446.json'

tags = []
with open(model_structure_459_14, 'r') as model_json_459_14, \
        open(model_structure_345_134, 'r') as model_json_345_134, \
        open(model_structure_600_446, 'r') as model_json_600_446:

    # 224x224
    model_459_14 = model_from_json(json.loads(model_json_459_14.read()))
    model_459_14.load_weights(weights_path_459_14)

    # 224x224
    model_345_134 = model_from_json(json.loads(model_json_345_134.read()))
    model_345_134.load_weights(weights_path_345_134)

    # 224x224
    model_600_446 = model_from_json(json.loads(model_json_600_446.read()))
    model_600_446.load_weights(weights_path_600_446)

    print 'models are loaded!'

    # loading the data
    count = 0
    result = []
    print 'start testing...'

    for f in common.iterate_minibatches(X_test, batchsize=1):
        test_batch_inputs = []

        img = cv2.imread('resource/test-jpg/{}'.format(f[0]))

        img = cv2.resize(img, (224, 224)).astype(np.float32)
        img = img.transpose((2, 0, 1))

        inputs = img

        test_batch_inputs.append(inputs)
        test_batch_inputs = common.aug(test_batch_inputs, inputs)

        test_batch_inputs = np.array(test_batch_inputs).astype(np.float32)

        predict_600_446 = model_600_446.predict_on_batch(test_batch_inputs)
        result_600_446 = common.agg(predict_600_446)
        result_600_446 = list(np.array(result_600_446).transpose() > thres_600_446)

        predict_459_14 = model_459_14.predict_on_batch(test_batch_inputs)
        result_459_14 = common.agg(predict_459_14)
        result_459_14 = list(np.array(result_459_14).transpose() > thres_459_14)

        predict_345_134 = model_345_134.predict_on_batch(test_batch_inputs)
        result_345_134 = common.agg(predict_345_134)
        result_345_134 = list(np.array(result_345_134).transpose() > thres_345_134)

        result = common.ensemble(np.array([result_600_446, result_459_14, result_345_134]))

        # print result1
        # print result2
        # print result

        t = [inv_label_map[i] for i, j in enumerate(result) if j]
        tags.append(' '.join(t))

        count += 1

        print '{}/{} predicted'.format(count, len(X_test))


df_test['tags'] = tags
df_test.to_csv('submission_0.csv', index=False)

print('\n{:.2f}m Runtime'.format((time.time() - st_time) / 60))
print '====== End ======'

