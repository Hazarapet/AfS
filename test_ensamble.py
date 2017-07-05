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

df_test = pd.DataFrame([[p.replace('.jpg', ''), p] for p in X_test])
df_test.columns = ['image_name', 'tags']

weights_path1 = 'models/nm/structures/tr_l:0.1162-tr_f2:0.9084-val_l:0.133-val_f2:0.8983-time:05-07-2017-00:25:12-dur:459.14.h5'
model_structure1 = 'models/nm/structures/tr_l:0.1162-tr_f2:0.9084-val_l:0.133-val_f2:0.8983-time:05-07-2017-00:25:12-dur:459.14.json'

weights_path2 = 'models/nm/structures/tr_l:0.1339-tr_f2:0.8899-val_l:0.1387-val_f2:0.8868-time:04-07-2017-13:50:07-dur:345.134.h5'
model_structure2 = 'models/nm/structures/tr_l:0.1339-tr_f2:0.8899-val_l:0.1387-val_f2:0.8868-time:04-07-2017-13:50:07-dur:345.134.json'

tags = []
with open(model_structure1, 'r') as model_json1, open(model_structure2, 'r') as model_json2:
    model1 = model_from_json(json.loads(model_json1.read()))
    model1.load_weights(weights_path1)

    model2 = model_from_json(json.loads(model_json2.read()))
    model2.load_weights(weights_path2)
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

        predict1 = model1.predict_on_batch(test_batch_inputs)
        result1 = common.agg(predict1)
        result = list(np.array(result1).transpose() > thres_459_14)

        # predict2 = model2.predict_on_batch(test_batch_inputs)
        # result2 = common.agg(predict2)
        # result2 = list(np.array(result2).transpose() > thres_345_134)

        # result = common.ensemble(np.array([result1, result2]))

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

