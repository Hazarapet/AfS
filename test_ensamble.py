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
thres_473_778 = [0.15, 0.17, 0.44, 0.14, 0.41, 0.63, 0.2, 0.18, 0.28, 0.27, 0.34, 0.2, 0.2, 0.09, 0.23, 0.2, 0.32]
thres_600_446 = [0.14, 0.15, 0.14, 0.27, 0.1, 0.35, 0.17, 0.32, 0.24, 0.12, 0.08, 0.1, 0.12, 0.13, 0.18, 0.16, 0.31]
thres_386_612 = [0.1, 0.13, 0.43, 0.21, 0.25, 0.47, 0.14, 0.63, 0.29, 0.19, 0.15, 0.17, 0.24, 0.1, 0.29, 0.13, 0.39]

thres_ens = [0.12, 0.16, 0.36, 0.28, 0.33, 0.45, 0.17, 0.62, 0.31, 0.14, 0.17, 0.19, 0.24, 0.09, 0.23, 0.17, 0.21]

df_test = pd.DataFrame([[p.replace('.jpg', ''), p] for p in X_test])
df_test.columns = ['image_name', 'tags']

# 0.91917
weights_path_459_14 = 'models/nm/structures/tr_l:0.1162-tr_f2:0.9084-val_l:0.133-val_f2:0.8983-time:05-07-2017-00:25:12-dur:459.14.h5'
model_structure_459_14 = 'models/nm/structures/tr_l:0.1162-tr_f2:0.9084-val_l:0.133-val_f2:0.8983-time:05-07-2017-00:25:12-dur:459.14.json'

# 0.93425 (whole db) ~0.921
weights_path_473_778 = 'models/nm/structures/tr_l:0.0876-tr_f2:0.9138-val_l:0.0785-val_f2:0.9242-time:07-07-2017-21:57:54-dur:473.778.h5'
model_structure_473_778 = 'models/nm/structures/tr_l:0.0876-tr_f2:0.9138-val_l:0.0785-val_f2:0.9242-time:07-07-2017-21:57:54-dur:473.778.json'

# 0.91892
weights_path_600_446 = 'models/nm/structures/tr_l:0.0912-tr_f2:0.9101-val_l:0.0969-val_f2:0.9071-time:07-07-2017-07:43:25-dur:600.446.h5'
model_structure_600_446 = 'models/nm/structures/tr_l:0.0912-tr_f2:0.9101-val_l:0.0969-val_f2:0.9071-time:07-07-2017-07:43:25-dur:600.446.json'

# 0.92475
weights_path_386_612 = 'models/nm/structures/tr_l:0.1016-tr_f2:0.9143-val_l:0.1081-val_f2:0.9094-time:12-07-2017-03:42:50-dur:386.612.h5'
model_structure_386_612 = 'models/nm/structures/tr_l:0.1016-tr_f2:0.9143-val_l:0.1081-val_f2:0.9094-time:12-07-2017-03:42:50-dur:386.612.json'

tags = []
with open(model_structure_459_14, 'r') as model_json_459_14, \
        open(model_structure_473_778, 'r') as model_json_473_778, \
        open(model_structure_386_612, 'r') as model_json_386_612, \
        open(model_structure_600_446, 'r') as model_json_600_446:

    # 224x224
    model_459_14 = model_from_json(json.loads(model_json_459_14.read()))
    model_459_14.load_weights(weights_path_459_14)

    # 224x224
    model_600_446 = model_from_json(json.loads(model_json_600_446.read()))
    model_600_446.load_weights(weights_path_600_446)

    # 224x224
    model_473_778 = model_from_json(json.loads(model_json_473_778.read()))
    model_473_778.load_weights(weights_path_473_778)

    # 256x256
    model_386_612 = model_from_json(json.loads(model_json_386_612.read()))
    model_386_612.load_weights(weights_path_386_612)

    print 'models are loaded!'

    # loading the data
    count = 0
    result = []
    print 'start testing...'

    for f in common.iterate_minibatches(X_test, batchsize=1):
        test_batch_inputs_224 = []
        test_batch_inputs_256 = []

        img = cv2.imread('resource/test-jpg/{}'.format(f[0]))

        img_224 = cv2.resize(img, (224, 224)).astype(np.float32)
        img_224 = img_224.transpose((2, 0, 1))

        img_256 = cv2.resize(img, (256, 256)).astype(np.float32)
        img_256 = img_256.transpose((2, 0, 1))

        inputs_224 = img_224
        inputs_256 = img_256

        test_batch_inputs_224.append(inputs_224)
        test_batch_inputs_224 = common.tta(test_batch_inputs_224, inputs_224)

        test_batch_inputs_256.append(inputs_256)
        test_batch_inputs_256 = common.tta(test_batch_inputs_256, inputs_256)

        test_batch_inputs_224 = np.array(test_batch_inputs_224).astype(np.float32)
        test_batch_inputs_256 = np.array(test_batch_inputs_256).astype(np.float32)

        #
        predict_600_446 = model_600_446.predict_on_batch(test_batch_inputs_224)
        result_600_446 = common.agg(predict_600_446)
        # result_600_446 = list(np.array(result_600_446).transpose() > thres_600_446)

        predict_459_14 = model_459_14.predict_on_batch(test_batch_inputs_224)
        result_459_14 = common.agg(predict_459_14)
        # result_459_14 = list(np.array(result_459_14).transpose() > thres_459_14)

        predict_473_778 = model_473_778.predict_on_batch(test_batch_inputs_224)
        result_473_778 = common.agg(predict_473_778)
        # result_473_778 = list(np.array(result_473_778).transpose() > thres_473_778)

        predict_386_612 = model_386_612.predict_on_batch(test_batch_inputs_256)
        result_386_612 = common.agg(predict_386_612)
        # result = list(np.array(result_386_612).transpose() > thres_386_612)

        r_600_446 = 1. * np.array(result_600_446)
        r_473_778 = 3. * np.array(result_473_778)
        r_459_14 = 2. * np.array(result_459_14)
        r_386_612 = 5. * np.array(result_386_612)

        result = np.sum([r_600_446, r_473_778, r_459_14, r_386_612], axis=0) / 11.
        result = list(np.array(result).transpose() > thres_ens)
        # Weighing the results
        # result = common.ensemble(np.array([result_600_446, result_459_14, result_473_778, result_473_778, result_386_612, result_386_612, result_386_612]))

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

